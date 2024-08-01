# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# DPT head for ViTs
# --------------------------------------------------------
# References: 
# https://github.com/isl-org/DPT
# https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/output_adapters.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, Tuple, Iterable, List, Optional, Dict
from networks.conv4d import Conv4d_Module
from networks.croco.cats import TransformerAggregator
from networks.croco.pose_regress import CrossBlock
from timm.models.layers import DropPath, trunc_normal_

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList([
        scratch.layer1_rn,
        scratch.layer2_rn,
        scratch.layer3_rn,
        scratch.layer4_rn,
    ])

    return scratch

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        width_ratio=1,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        self.width_ratio = width_ratio

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(res, size=(output.shape[2], output.shape[3]), mode='bilinear')

            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if self.width_ratio != 1:
            # and output.shape[3] < self.width_ratio * output.shape[2]
            #size=(image.shape[])
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output  = F.interpolate(output, size=(2* output.shape[2], shape), mode='bilinear')
        else:
            output = nn.functional.interpolate(output, scale_factor=2,
                    mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def make_fusion_block(features, use_bn, width_ratio=1):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        width_ratio=width_ratio,
    )

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

class DPTOutputAdapter(nn.Module):
    """DPT output adapter.

    :param num_cahnnels: Number of output channels
    :param stride_level: tride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param hooks: Index of intermediate layers
    :param layer_dims: Dimension of intermediate layers
    :param feature_dim: Feature dimension
    :param last_dim: out_channels/in_channels for the last two Conv2d when head_type == regression
    :param use_bn: If set to True, activates batch norm
    :param dim_tokens_enc:  Dimension of tokens coming from encoder
    """

    def __init__(self,
                 num_channels: int = 1,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 main_tasks: Iterable[str] = ('rgb',),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 last_dim: int = 32,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 output_width_ratio=1,
                 max_depth = 80.,
                 residual = False,
                 single = False,
                 args = None,
                 with_pose = False,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type
        self.max_depth=max_depth
        self.residual = residual
        self.single = single
        self.args = args
        self.with_pose = with_pose

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        
        if self.with_pose:
            self.pose_agg = CrossBlock(dim=768)
            self.pose_regressor = nn.Sequential(
                nn.Linear((768+6), 512 ),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                
            )
            self.rotation_regressor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 6),
            )
            self.translation_regressor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
        
        
        
        
        if self.args.attn_conv4d:
            self.norm0 = nn.BatchNorm2d(256+480)
            self.norm1 = nn.BatchNorm2d(256+480)
            self.norm2 = nn.BatchNorm2d(256+480)
            self.norm3 = nn.BatchNorm2d(256+480)

            self.aggregator0 = nn.Sequential(nn.GELU(),
                                    nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            self.aggregator1 = nn.Sequential(nn.GELU(),
                                            nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),                            
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            self.aggregator2 = nn.Sequential(nn.GELU(),
                                            nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            self.aggregator3 = nn.Sequential(nn.GELU(),
                                            nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))


        if self.head_type == 'regression':

            
            self.conv_disp3= nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                                nn.Sigmoid(),
                                )
            
            self.conv_disp2= nn.Sequential( 
                            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                            nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(True),
                            nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                            nn.Sigmoid(),
                                            )
            
            self.conv_disp1= nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                                nn.Sigmoid(),
                                            )
            
            self.conv_disp0= nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                                nn.Sigmoid(),
                                            )
        elif self.head_type == 'semseg':
            # The "DPTSegmentationModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(feature_dim, self.num_channels, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            raise ValueError('DPT head_type must be "regression" or "semseg".')

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

        if self.args.attn_conv4d:
            self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.aggregator0[1].weight, std=.02)
        trunc_normal_(self.aggregator0[2].weight, std=.02)
        trunc_normal_(self.aggregator1[1].weight, std=.02)
        trunc_normal_(self.aggregator1[2].weight, std=.02)
        trunc_normal_(self.aggregator2[1].weight, std=.02)
        trunc_normal_(self.aggregator2[2].weight, std=.02)
        trunc_normal_(self.aggregator3[1].weight, std=.02)
        trunc_normal_(self.aggregator3[2].weight, std=.02)

        nn.init.constant_(self.aggregator0[1].bias, 0)
        nn.init.constant_(self.aggregator0[2].bias, 0)
        nn.init.constant_(self.aggregator1[1].bias, 0)
        nn.init.constant_(self.aggregator1[2].bias, 0)
        nn.init.constant_(self.aggregator2[1].bias, 0)
        nn.init.constant_(self.aggregator2[2].bias, 0)
        nn.init.constant_(self.aggregator3[1].bias, 0)
        nn.init.constant_(self.aggregator3[2].bias, 0)

        nn.init.constant_(self.norm0.bias, 0)
        nn.init.constant_(self.norm1.bias, 0)
        nn.init.constant_(self.norm2.bias, 0)
        nn.init.constant_(self.norm3.bias, 0)

        nn.init.constant_(self.norm0.weight, 1.0)
        nn.init.constant_(self.norm1.weight, 1.0)
        nn.init.constant_(self.norm2.weight, 1.0)
        nn.init.constant_(self.norm3.weight, 1.0)



    def init(self, dim_tokens_enc=768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        #print(dim_tokens_enc)

        # Set up activation postprocessing layers
        if isinstance(dim_tokens_enc, int):
            dim_tokens_enc = 4 * [dim_tokens_enc]

        self.dim_tokens_enc = [dt * len(self.main_tasks) for dt in dim_tokens_enc]

        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[0],
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[1],
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[2],
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])

    def adapt_tokens(self, encoder_tokens):
        # Adapt tokens
        x = []
        x.append(encoder_tokens[:, :])
        x = torch.cat(x, dim=-1)
        return x
    
    def r6d2mat(self,d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalisation per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
                first two rows of the rotation matrix. 
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row

    def forward(self, encoder_tokens: List[torch.Tensor], image_size, attn_map=None, intrinsics=None):
            #input_info: Dict):
        outputs={}
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = image_size
        
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]
        
        if self.residual:
            layers = [layers[0]+layers[4], layers[1]+layers[5], layers[2]+layers[6], layers[3]+layers[7]]
        
        if self.with_pose:
            layers_tmp = [layer.detach().clone() for layer in layers]
            layers_tmp = torch.stack(layers_tmp,dim=1).mean(dim=1)
        
        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]
        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]
        

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        
        if self.args.attn_conv4d:
            resi = 0
            attn_maps = [(attn_map[hook-12] + attn_map[hook-13] + attn_map[hook-14])/3. for hook in self.hooks]
            attn_maps = [rearrange(l.mean(dim=1), 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in attn_maps]
            attn_sizes = [(N_H//2,N_W//2),(N_H,N_W),(N_H*2,N_W*2),(N_H*4,N_W*4)]
            
            attn_maps[0] = F.interpolate(attn_maps[0], size=attn_sizes[3], mode='bilinear', align_corners=False)
            attn_maps[1] = F.interpolate(attn_maps[1], size=attn_sizes[2], mode='bilinear', align_corners=False)
            attn_maps[2] = F.interpolate(attn_maps[2], size=attn_sizes[1], mode='bilinear', align_corners=False)
            attn_maps[3] = F.interpolate(attn_maps[3], size=attn_sizes[0], mode='bilinear', align_corners=False)
            
            input0 = torch.cat([layers[resi+0], attn_maps[0]], dim=1)
            input1 = torch.cat([layers[resi+1], attn_maps[1]], dim=1)
            input2 = torch.cat([layers[resi+2], attn_maps[2]], dim=1)
            input3 = torch.cat([layers[resi+3], attn_maps[3]], dim=1)
            
            
            layers[resi+3] = self.aggregator3(input3) + layers[resi+3]
            layers[resi+2] = self.aggregator2(input2) + layers[resi+2]
            layers[resi+1] = self.aggregator1(input1) + layers[resi+1]
            layers[resi+0] = self.aggregator0(input0) + layers[resi+0]

            

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])
        
        if self.with_pose:
            layers_0 = rearrange(layers[0],'b c nh nw -> b (nh nw) c')
            B = layers_0.shape[0]
            
            pose_feat_ctxt = self.pose_agg(layers_tmp.detach(),corr=torch.stack(attn_map,dim=1).mean(dim=1).mean(dim=1),intrinsics=intrinsics).mean(dim=2)
            pose_latent_ctxt = self.pose_regressor(pose_feat_ctxt)
            
            rot_ctxt, tran_ctxt = self.rotation_regressor(pose_latent_ctxt), self.translation_regressor(pose_latent_ctxt)# Bxn_views x 9, Bxn_views x 3 
            R_ctxt = self.r6d2mat(rot_ctxt)[:, :3, :3] 

            estimated_rel_pose_ctxt = torch.cat((torch.cat((R_ctxt, tran_ctxt.unsqueeze(-1)), dim=-1),torch.FloatTensor([0,0,0,1]).expand(B,1,-1).to(tran_ctxt.device)), dim=1) #estimated pose between query and context 2
            outputs['pose'] = estimated_rel_pose_ctxt
        

        # Output head
        # out = self.head(path_1)
        outputs['pred_disp',3] = self.conv_disp3(path_4)    # (b,1,12,40)   # passed through sigmoid. [0~1]
        outputs['pred_disp',2] = self.conv_disp2(path_3)    # (b,1,24,80)
        outputs['pred_disp',1] = self.conv_disp1(path_2)    # (b,1,48,160)
        outputs['pred_disp',0] = self.conv_disp0(path_1)    # (b,1,96,320)

        return outputs
    
    
    
    
class DPTOutputAggregateAdapter(nn.Module):
    """DPT output adapter.

    :param num_cahnnels: Number of output channels
    :param stride_level: tride level compared to the full-sized image.
        E.g. 4 for 1/4th the size of the image.
    :param patch_size_full: Int or tuple of the patch size over the full image size.
        Patch size for smaller inputs will be computed accordingly.
    :param hooks: Index of intermediate layers
    :param layer_dims: Dimension of intermediate layers
    :param feature_dim: Feature dimension
    :param last_dim: out_channels/in_channels for the last two Conv2d when head_type == regression
    :param use_bn: If set to True, activates batch norm
    :param dim_tokens_enc:  Dimension of tokens coming from encoder
    """

    def __init__(self,
                 num_channels: int = 1,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 main_tasks: Iterable[str] = ('rgb',),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 last_dim: int = 32,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 output_width_ratio=1,
                 max_depth = 80.,
                 with_pose = False,
                 residual = False,
                 args = None,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type
        self.max_depth=max_depth
        self.residual = residual
        self.args = args

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        # self.feature_proj = nn.ModuleList([nn.Linear(256, 128) for i in range(4)])

        self.aggregator0 = nn.Sequential( nn.GELU(),
                                          nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+480, kernel_size=3, stride=1, padding=1))
        self.aggregator1 = nn.Sequential(nn.GELU(),
                                        nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+480, kernel_size=3, stride=1, padding=1))
        self.aggregator2 = nn.Sequential( nn.GELU(),
                                          nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+480, kernel_size=3, stride=1, padding=1))
        self.aggregator3 = nn.Sequential( nn.GELU(),
                                          nn.Conv2d(256+480, 256, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(256, 256+480, kernel_size=3, stride=1, padding=1))
        
        self.proj = nn.ModuleList([nn.Conv2d(256+480, 480,  kernel_size=1, stride=1, padding=0) for i in range(4)])
        
        # self.depth_head0 = nn.Sequential(ResidualConvUnit_custom(480,nn.ReLU(),False),
        #                                 nn.Conv2d(480, feature_dim, kernel_size=1, stride=1, padding=0))
        # self.depth_head1 = nn.Sequential(ResidualConvUnit_custom(480,nn.ReLU(),False),
        #                         nn.Conv2d(480, feature_dim, kernel_size=1, stride=1, padding=0))
        # self.depth_head2 = nn.Sequential(ResidualConvUnit_custom(480,nn.ReLU(),False),
        #                 nn.Conv2d(480, feature_dim, kernel_size=1, stride=1, padding=0))
        # self.depth_head3 = nn.Sequential(ResidualConvUnit_custom(480,nn.ReLU(),False),
        #             nn.Conv2d(480, feature_dim, kernel_size=1, stride=1, padding=0))
        
        self.depth_head0 = nn.Sequential(nn.Conv2d(480, feature_dim, kernel_size=3, stride=1, padding=1))
        self.depth_head1 = nn.Sequential(nn.Conv2d(480, feature_dim, kernel_size=3, stride=1, padding=1))
        self.depth_head2 = nn.Sequential(nn.Conv2d(480, feature_dim, kernel_size=3, stride=1, padding=1))
        self.depth_head3 = nn.Sequential(nn.Conv2d(480, feature_dim, kernel_size=3, stride=1, padding=1))
        
        self.with_pose = with_pose
        if self.with_pose:
            self.pose_agg = CrossBlock()
            
            self.pose_regressor = nn.Sequential(

                nn.Linear((16*16+6)*256, 512 ),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                
            )
            self.rotation_regressor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 6),
            )
            self.translation_regressor = nn.Sequential(
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )

        if self.head_type == 'regression':
            self.conv_disp3= nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                                nn.Sigmoid(),
                                )
            
            self.conv_disp2= nn.Sequential( 
                            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                            nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                            nn.ReLU(True),
                            nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                            nn.Sigmoid(),
                                            )
            
            self.conv_disp1= nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                                nn.Sigmoid(),
                                            )
            
            self.conv_disp0= nn.Sequential(
                                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(True),
                                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0),
                                nn.Sigmoid(),
                                            )
        elif self.head_type == 'semseg':
            # The "DPTSegmentationModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(feature_dim, self.num_channels, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            raise ValueError('DPT head_type must be "regression" or "semseg".')

        # if self.dim_tokens_enc is not None:
        #     self.init(dim_tokens_enc=dim_tokens_enc)


    def init(self, dim_tokens_enc=768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        #print(dim_tokens_enc)

        # Set up activation postprocessing layers
        if isinstance(dim_tokens_enc, int):
            dim_tokens_enc = 4 * [dim_tokens_enc]

        self.dim_tokens_enc = [dt * len(self.main_tasks) for dt in dim_tokens_enc]

        self.act_1_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[0],
                out_channels=self.layer_dims[0],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[1],
                out_channels=self.layer_dims[1],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[2],
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.dim_tokens_enc[3],
                out_channels=self.layer_dims[3],
                kernel_size=1, stride=1, padding=0,
            ),
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )

        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])

    def adapt_tokens(self, encoder_tokens):
        # Adapt tokens
        x = []
        x.append(encoder_tokens[:, :])
        x = torch.cat(x, dim=-1)
        return x
    
    def r6d2mat(self,d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalisation per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
                first two rows of the rotation matrix. 
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row

    def forward(self, encoder_tokens: List[torch.Tensor], image_size, attn_map,intrinsics=None):
            #input_info: Dict:
        outputs={}
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = image_size
        
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]
        if self.residual:
            layers = [layers[0]+layers[4], layers[1]+layers[5], layers[2]+layers[6], layers[3]+layers[7]]
        attn_maps = [(attn_map[hook-12] + attn_map[hook-13] + attn_map[hook-14])/3. for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]
        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]
        
        attn_maps = [rearrange(l.mean(dim=1), 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in attn_maps]
        attn_sizes = [(6,20),(12,40),(24,80),(48,160)]
        # attn_maps = [rearrange(l, 'b c nh nw -> b (nh nw) c', nh=attn_sizes[idx][0], nw=attn_sizes[idx][1]) for idx, l in enumerate(attn_maps)]

        # layer3 = self.feature_proj[3](rearrange(layers[3], 'b c nh nw -> b (nh nw) c')).unsqueeze(dim=1)
        
        attn_map3 = F.interpolate(attn_maps[3], size=attn_sizes[0], mode='bilinear')
        attn_input3 = torch.cat([attn_map3, layers[3]], dim=1)
        attn3_out = self.aggregator3(attn_input3) + attn_input3
        attn3_out = self.proj[3](attn3_out)
        
        attn2 = F.interpolate(attn3_out, size=attn_sizes[1], mode='bilinear')
        attn_maps[2] = F.interpolate(attn_maps[2], size=attn_sizes[1], mode='bilinear')
        attn2 = attn2 + attn_maps[2]
        
        attn_input2 = torch.cat([attn2, layers[2]], dim=1)
        attn2_out = self.aggregator1(attn_input2) + attn_input2
        attn2_out = self.proj[2](attn2_out)

        attn1 = F.interpolate(attn2_out, size=attn_sizes[2], mode='bilinear')
        attn_maps[1] = F.interpolate(attn_maps[1], size=attn_sizes[2], mode='bilinear')
        attn1 = attn1 + attn_maps[1]
        
        attn_input1 = torch.cat([attn1, layers[1]], dim=1)
        attn1_out = self.aggregator1(attn_input1) + attn_input1
        attn1_out = self.proj[1](attn1_out)
        
        
        attn0 = F.interpolate(attn1_out, size=attn_sizes[3], mode='bilinear')
        attn_maps[0] = F.interpolate(attn_maps[0], size=attn_sizes[3], mode='bilinear')
        attn0 = attn0 + attn_maps[0]
        
        attn_input0 = torch.cat([attn0, layers[0]], dim=1)
        attn0_out = self.aggregator3(attn_input0) + attn_input0
        attn0_out = self.proj[0](attn0_out)
        
        path_4 = self.depth_head3(attn3_out)
        path_3 = self.depth_head2(attn2_out)
        path_2 = self.depth_head1(attn1_out)
        path_1 = self.depth_head0(attn0_out)
        
        if self.with_pose:
            layers_0 = rearrange(layers[0],'b c nh nw -> b (nh nw) c')
            B = layers_0.shape[0]
            pose_feat_ctxt = self.pose_agg(layers_0,corr=attn3,intrinsics=intrinsics).reshape(B,-1)
            pose_latent_ctxt = self.pose_regressor(pose_feat_ctxt)
            
            rot_ctxt, tran_ctxt = self.rotation_regressor(pose_latent_ctxt), self.translation_regressor(pose_latent_ctxt)# Bxn_views x 9, Bxn_views x 3 
            R_ctxt = self.r6d2mat(rot_ctxt)[:, :3, :3] 

            estimated_rel_pose_ctxt = torch.cat((torch.cat((R_ctxt, tran_ctxt.unsqueeze(-1)), dim=-1),torch.FloatTensor([0,0,0,1]).expand(B,1,-1).to(tran_ctxt.device)), dim=1) #estimated pose between query and context 2
            outputs['pose'] = estimated_rel_pose_ctxt
        # Output head
        # out = self.head(path_1)
        outputs['pred_disp',3] = self.conv_disp3(path_4)    # (b,1,12,40)   # passed through sigmoid. [0~1]
        outputs['pred_disp',2] = self.conv_disp2(path_3)    # (b,1,24,80)
        outputs['pred_disp',1] = self.conv_disp1(path_2)    # (b,1,48,160)
        outputs['pred_disp',0] = self.conv_disp0(path_1)    # (b,1,96,320)

        return outputs