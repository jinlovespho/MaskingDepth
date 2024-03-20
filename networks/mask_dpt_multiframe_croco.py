import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random
import numpy as np

from .croco_blocks import *

class Masked_DPT_Multiframe_Croco(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        max_depth,
        features=[96, 192, 384, 768],
        hooks=[2, 5, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
        num_prev_frame=1,
        masking_ratio=0.5,
        num_frame_to_mask=1,
        cross_attn_depth = 8
    ):
        super().__init__()
        
        # JINLOVESPHO
        self.num_prev_frame=num_prev_frame
        
        # ViT
        self.encoder = encoder
        self.encoder.transformer.set_hooks(hooks)
        self.hooks = hooks

        #read out processing (ignore / add / project[dpt use this process])
        readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)

        # 32, 48, 136, 384
        self.act_postprocess1 = nn.Sequential(
            # readout_oper[0],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,       # vit_features = d_model = embedding dim 의미 
                out_channels=features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[0],
                out_channels=features[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.act_postprocess2 = nn.Sequential(
            # readout_oper[1],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=features[1],
                out_channels=features[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )
        self.act_postprocess3 = nn.Sequential(
            # readout_oper[2],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[2],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )
        self.act_postprocess4 = nn.Sequential(
            # readout_oper[3],
            Transpose(1, 2),
            nn.Conv2d(
                in_channels=vit_features,
                out_channels=features[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=features[3],
                out_channels=features[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )
        
        self.unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(                                                                 # get_image_size()= (192,640)
                    [                                                                       # get_patch_size()= (16,16)
                        encoder.get_image_size()[0] // encoder.get_patch_size()[0],         # 즉 이 과정은 하나의 
                        encoder.get_image_size()[1] // encoder.get_patch_size()[1],
                    ]
                ),
            )
        )

        self.scratch = make_scratch(features, 256)
        self.scratch.refinenet1 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet2 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet3 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet4 = make_fusion_block(features=256, use_bn=False)

        self.scratch.output_conv = head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            nn.Identity(),
        )

        self.max_depth = max_depth
        self.target_size = encoder.get_image_size()
        
        # JINLOVESPHO
        self.masking_ratio=masking_ratio
        self.num_frame_to_mask = num_frame_to_mask
        
        # only make one mask token 
        self.msk_tkn1 = nn.Parameter(torch.randn(vit_features))
        self.msk_tkn2 = nn.Parameter(torch.randn(vit_features))
        self.msk_tkn3 = nn.Parameter(torch.randn(vit_features))
        self.msk_tkn4 = nn.Parameter(torch.randn(vit_features))


        # self.mask_pe_table = nn.Embedding(encoder.num_patches, vit_features)
        self.decoder_pose_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(vit_features, self.target_size[0]//16,self.target_size[1]//16, 0)).float(), requires_grad=False)
        
        self.cross_attn_module1 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None)
        self.cross_attn_module2 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None)
        self.cross_attn_module3 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None)
        self.cross_attn_module4 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None)
        
    
    def forward(self, img_frames, K = 1, mode=None):

        # tokenize input image frames(t,t-1, . . ) and add positional embeddings
        tokenized_frames = []
        for i, frame in enumerate(img_frames):
            tmp =self.encoder.to_patch_embedding(frame)
            tmp += self.encoder.pos_emb_lst[i][:,1:,:]
            tmp = self.encoder.dropout(tmp)
            tokenized_frames.append(tmp)
        
        # batch_size, length, dim, device
        b,n,dim = tokenized_frames[0].shape
        device = tokenized_frames[0].device.type  
        
        # number of patches to mask
        num_p_msk = int(self.masking_ratio * n )

        # random masking index generation
        idx_rnd = torch.rand(b,n, device=device).argsort()
        idx_msk, idx_umsk = idx_rnd[:,:num_p_msk], idx_rnd[:,num_p_msk:]
        idx_msk = idx_msk.sort().values
        idx_umsk = idx_umsk.sort().values
        idx_bs = torch.arange(b)[:,None]
        
        # unmasked tokens
        frame0_unmsk_tkn = tokenized_frames[0][idx_bs, idx_umsk]

        # masked tokens
        msk_tkns1 = repeat(self.msk_tkn1, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns2 = repeat(self.msk_tkn2, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns3 = repeat(self.msk_tkn3, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns4 = repeat(self.msk_tkn4, 'd -> b n d', b=b, n=num_p_msk)

        # pos_msk_tkns = self.mask_pe_table(idx_msk)
        # pos_umsk_tkns = self.mask_pe_table(idx_umsk)
        
        # add positional embedding for masked tokens
        
        
        # t frame encoder
        glob1 = self.encoder.transformer(frame0_unmsk_tkn)   
        glob1_layer_1 = self.encoder.transformer.features[0]      
        glob1_layer_2 = self.encoder.transformer.features[1]      
        glob1_layer_3 = self.encoder.transformer.features[2]      
        glob1_layer_4 = self.encoder.transformer.features[3]      
                    
        # t-1 frame encoder
        glob2 = self.encoder.transformer(tokenized_frames[1])     
        glob2_layer_1 = self.encoder.transformer.features[0]      
        glob2_layer_2 = self.encoder.transformer.features[1]      
        glob2_layer_3 = self.encoder.transformer.features[2]      
        glob2_layer_4 = self.encoder.transformer.features[3]      


        frame0_msk_umsk_tkn1 = torch.zeros(b, n, dim, device=device)
        frame0_msk_umsk_tkn1[idx_bs, idx_umsk] = glob1_layer_1
        frame0_msk_umsk_tkn1[idx_bs, idx_msk] = msk_tkns1
        frame0_msk_umsk_tkn1 = frame0_msk_umsk_tkn1 + self.decoder_pose_embed
        glob2_layer_1 = glob2_layer_1 + self.decoder_pose_embed
        
        frame0_msk_umsk_tkn2 = torch.zeros(b, n, dim, device=device)
        frame0_msk_umsk_tkn2[idx_bs, idx_umsk] = glob1_layer_2
        frame0_msk_umsk_tkn2[idx_bs, idx_msk] = msk_tkns2
        frame0_msk_umsk_tkn2 = frame0_msk_umsk_tkn2 + self.decoder_pose_embed
        glob2_layer_2 = glob2_layer_2 + self.decoder_pose_embed
        
        frame0_msk_umsk_tkn3 = torch.zeros(b, n, dim, device=device)
        frame0_msk_umsk_tkn3[idx_bs, idx_umsk] = glob1_layer_3 
        frame0_msk_umsk_tkn3[idx_bs, idx_msk] = msk_tkns3
        frame0_msk_umsk_tkn3 = frame0_msk_umsk_tkn3 + self.decoder_pose_embed
        glob2_layer_3 = glob2_layer_3 + self.decoder_pose_embed

        frame0_msk_umsk_tkn4 = torch.zeros(b, n, dim, device=device)
        frame0_msk_umsk_tkn4[idx_bs, idx_umsk] = glob1_layer_4
        frame0_msk_umsk_tkn4[idx_bs, idx_msk] = msk_tkns4
        frame0_msk_umsk_tkn4 = frame0_msk_umsk_tkn4 + self.decoder_pose_embed
        glob2_layer_4 = glob2_layer_4 + self.decoder_pose_embed

        cross_attn_out1 = self.cross_attn_module1(frame0_msk_umsk_tkn1, glob2_layer_1) 
        cross_attn_out2 = self.cross_attn_module2(frame0_msk_umsk_tkn2, glob2_layer_2)
        cross_attn_out3 = self.cross_attn_module3(frame0_msk_umsk_tkn3, glob2_layer_3)
        cross_attn_out4 = self.cross_attn_module4(frame0_msk_umsk_tkn4, glob2_layer_4)
                    
        # JINLOVESPHO
        layer_1 = cross_attn_out1
        layer_2 = cross_attn_out2
        layer_3 = cross_attn_out3
        layer_4 = cross_attn_out4
        
        layer_1 = self.act_postprocess1[0](layer_1)     # 모두 index[0] 이므로, Tranpose() 통과 의미.
        layer_2 = self.act_postprocess2[0](layer_2)     # 즉 모두(B, 480, 768) -> Transpose -> (B, 768, 480) 된다.
        layer_3 = self.act_postprocess3[0](layer_3)
        layer_4 = self.act_postprocess4[0](layer_4)


        # transformer encoder transposed outputs (intermediate ouputs 포함 O)
        features= [layer_1, layer_2, layer_3, layer_4]

        # 이제 transformer의 output은 2D 꼴이다. (batch 생략 경우)
        # transformer output shape = (B, 480, 768)
        # 이것을 "CNN" decoder에 넣기 위해 3D 로 reshape 해주어야 
        # 각 patch를 하나의 pixel로 취급하면
        # (480,768) = (N,D) -> (C,H,W) 꼴로 만들어야 
        # (480,768) -> (768, H, W) -> (768, height patch 개수, width patch 개수) -> (768, 192/16, 640/16) -> (768, 12, 40) 이 되는 것 ! 
        
        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1)       # (B,768,12,40) 
        if layer_2.ndim == 3:   
            layer_2 = self.unflatten(layer_2)       # (B,768,12,40)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3)       # (B,768,12,40)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4)       # (B,768,12,40)

        # 여기는 refinenet에 넣기 위해 여러 scale로 만들어 O
        layer_1 = self.act_postprocess1[1:](layer_1)    # channels 768 -> 96,  layer_1.shape: (B, 96, 48, 160) = (B,  C,    H,   W) 로 생각하면 이제 이해 O
        layer_2 = self.act_postprocess2[1:](layer_2)    # channels 768 -> 192, layer_2.shape: (B, 192, 24, 80) = (B, 2C, 1/2H, 1/2W)
        layer_3 = self.act_postprocess3[1:](layer_3)    # channels 768 -> 384, layer_3.shape: (B, 384, 12, 40) = (B, 4C, 1/4H, 1/4W)
        layer_4 = self.act_postprocess4[1:](layer_4)    # channels 768 -> 768, layer_4.shape: (B, 768, 6, 20) =  (B, 8C, 1/8H, 1/8W)

        # 아닌데.. 여기가 refinenet 에 넣기 전에 모든 channel 을 ? -> 256 으로 맞춰주는 conv layer들.
        layer_1_rn = self.scratch.layer1_rn(layer_1)    # channels 96 ->  256,  layer_1_rn.shape: (B, 256, 48, 160)
        layer_2_rn = self.scratch.layer2_rn(layer_2)    # channels 192 -> 256,  layer_2_rn.shape: (B, 256, 24, 80)
        layer_3_rn = self.scratch.layer3_rn(layer_3)    # channels 384 -> 256,  layer_3_rn.shape: (B, 256, 12, 40)
        layer_4_rn = self.scratch.layer4_rn(layer_4)    # channels 768 -> 256,  layer_4_rn.shape: (B, 256, 6, 20)
        
        fusion_features = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]


        # 여기가 refinenet 논문에 나오는 refinenet 이다.
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        pred_depth = self.scratch.output_conv(path_1) * self.max_depth
        
        # pred_depth = (B,1,192,640)
        # features = len() = 4 = transformer 의 중간 layer outputs, (B,N,D)
        # fusion_features len() = 4 = 위 transformer 의 중간 layer outputs 를 CNN refinenet에 넣기 위해 img shape으로 여러 scale로 reshape한 (B,C,H,W) 꼴 
        return pred_depth, features, fusion_features
    
    
    def resize_image_size(self, h, w, start_index=1):
        self.encoder.resize_pos_embed(h, w, start_index)
        self.unflatten = nn.Sequential( 
                            nn.Unflatten(
                                2,
                                torch.Size(
                                    [
                                        self.encoder.get_image_size()[0] // self.encoder.get_patch_size()[0],
                                        self.encoder.get_image_size()[1] // self.encoder.get_patch_size()[1],
                                    ]
                                ),
                            )   
                        )
    def target_out_size(self,h, w):
        self.unflatten = nn.Sequential(
            nn.Unflatten(
                2,
                torch.Size(
                    [
                        h // self.encoder.get_patch_size()[0],
                        w // self.encoder.get_patch_size()[1],
                    ]
                ),
            )
        )
        self.target_size = (h,w)
        

def get_2d_sincos_pos_embed(embed_dim, height,width, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(height, dtype=np.float32)
    grid_w = np.arange(width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, height, width])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb