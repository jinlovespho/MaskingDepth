import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random

class Masked_DPT(nn.Module):
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
    ):
        super().__init__()
        
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
                torch.Size(
                    [
                        encoder.get_image_size()[0] // encoder.get_patch_size()[0],
                        encoder.get_image_size()[1] // encoder.get_patch_size()[1],
                    ]
                ),
            )
        )

        self.scratch = make_scratch(features, 256)
        self.scratch.refinenet1 = make_fusion_block(256, False)
        self.scratch.refinenet2 = make_fusion_block(256, False)
        self.scratch.refinenet3 = make_fusion_block(256, False)
        self.scratch.refinenet4 = make_fusion_block(256, False)

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

    def forward(self, img, K = 1):
        # assert mask_ratio >= 0 and mask_ratio < 1, 'masking ratio must be kept between 0 and 1'

        breakpoint()
        
        # img.shape: (B, 3, 192, 640) = (B,3, 12*16, 40*16) 즉 patch 로 나누면 12개 * 40 개 patch 나온다.
        # patch_size= 16
        # 12*40 = 480 이 token 개수 
        
        x = self.encoder.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.encoder.pos_embedding[:, 1:(n + 1)]
        x = self.encoder.dropout(x)
        
        if K == 1:
            glob = self.encoder.transformer(x)

            layer_1 = self.encoder.transformer.features[0]      # features 가 뭐지. 앞에서 self.hooks = [2,5,8,11] 이므로
            layer_2 = self.encoder.transformer.features[1]      # transformer 의 2,5,8,11 th block 의 intermediate output values
            layer_3 = self.encoder.transformer.features[2]      # 를 layer_1,2,3,4 에 담은 것.
            layer_4 = self.encoder.transformer.features[3] 
            
            # self.act_postprocess1 구성 => Transpose(), Conv2d(), ConvTranspose2d()
            # self.act_postprocess2 구성 => Transpose(), Conv2d(), ConvTranpose2d()
            # self.act_postprocess3 구성 => Tranpose(), Conv2d()
            # self.act_postprocess4 구성 => Tranpose(), Conv2d(), Conv2d()
            
            layer_1 = self.act_postprocess1[0](layer_1)     # 모두 index[0] 이므로, Tranpose() 통과 의미.
            layer_2 = self.act_postprocess2[0](layer_2)     # 즉 모두(B, 480, 768) -> Transpose -> (B, 768, 480) 된다.
            layer_3 = self.act_postprocess3[0](layer_3)
            layer_4 = self.act_postprocess4[0](layer_4)

        else:
            num_patches = (self.target_size[0] // self.encoder.get_patch_size()[0]) * \
                          (self.target_size[1] // self.encoder.get_patch_size()[1])
            batch_range = torch.arange(b, device = x.device)[:, None]
            rand_indices = torch.rand(b, num_patches, device = x.device).argsort(dim = -1)

            ### random shuffle the patches
            x = x[batch_range,rand_indices]
           
            ### assign mask
            v = sorted([random.randint(1,num_patches-1) for i in range(int(K-1))] + [0, num_patches])
            mask_v = torch.zeros(len(v[:-1]), num_patches).to(x.device)
            for i in range(len(v[:-1])):
                mask_v[i, v[i]:v[i+1]] = 1.0

            ### K-way augmented attention
            partial_token = self.encoder.transformer(x, (mask_v.transpose(0,1) @ mask_v))
            reform_indices = torch.argsort(rand_indices, dim=1)

            
            #no class
            layer_1 = self.act_postprocess1[0](self.encoder.transformer.features[0][batch_range, reform_indices])
            layer_2 = self.act_postprocess2[0](self.encoder.transformer.features[1][batch_range, reform_indices])
            layer_3 = self.act_postprocess3[0](self.encoder.transformer.features[2][batch_range, reform_indices])
            layer_4 = self.act_postprocess4[0](self.encoder.transformer.features[3][batch_range, reform_indices])

        # transformer encoder outputs (intermediate ouputs 포함 O)
        features= [layer_1, layer_2, layer_3, layer_4]

        # 여기서 unflatten 은 head 기준으로 token을 나누는 것
        # (B, 480, 768) -> Transpose() -> (B,768,480) -> unflatten -> (B, 768, 12, 40) where 12=num_heads.
        if layer_1.ndim == 3:
            layer_1 = self.unflatten(layer_1)       
        if layer_2.ndim == 3:
            layer_2 = self.unflatten(layer_2)
        if layer_3.ndim == 3:
            layer_3 = self.unflatten(layer_3)
        if layer_4.ndim == 3:
            layer_4 = self.unflatten(layer_4)

        # 이 과정을 왜하는지 이해가 X. 기존 layer_i.shape = (B, 768, 12, 40) 을 왜 다 이상하게 바꾸는거지.
        # transformer 의 output 들을 refinenet 에 넣기 전에 뭔가를 해줄려는 것 같은데 . .
        breakpoint()
        layer_1 = self.act_postprocess1[1:](layer_1)    # channels 768 -> 96,  layer_1.shape: (B, 96, 48, 160)
        layer_2 = self.act_postprocess2[1:](layer_2)    # channels 768 -> 192, layer_2.shape: (B, 192, 24, 80)
        layer_3 = self.act_postprocess3[1:](layer_3)    # channels 768 -> 384, layer_3.shape: (B, 384, 12, 40)
        layer_4 = self.act_postprocess4[1:](layer_4)    # channels 768 -> 768, layer_4.shape: (B, 768, 6, 20)

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
        
