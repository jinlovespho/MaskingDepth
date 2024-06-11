import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random

class SF_Depth_Baseline(nn.Module):
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
        self.vit_features = vit_features 
        
        self.target_size = encoder.get_image_size() # hg
        self.img_h, self.img_w = self.encoder.get_image_size()
        self.pH, self.pW = self.encoder.get_patch_size()
        self.num_pH, self.num_pW = self.img_h//self.pH, self.img_w//self.pW
        
        self.max_depth = max_depth

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


    def forward(self, inputs, train_args, mode):
        outputs={}
        
        img_frames=[]
        if mode == 0:
            img_frames.append(inputs['color',  0, 0])   # curr_frame : [0]
            # img_frames.append(inputs['color', -1, 0])   # prev_frame : [1]
            # img_frames.append(inputs['color',  1, 0])   # future_frame: [2]
        else:
            img_frames.append(inputs['color_aug',  0, 0])   
            # img_frames.append(inputs['color_aug', -1, 0])   
            # img_frames.append(inputs['color_aug',  1, 0])   
        
        x = self.encoder.to_patch_embedding(img_frames[0])    # x=(B, 480,768): 480 token, each 768 dim
        b, n, _ = x.shape
        x += self.encoder.pos_embedding[:, 1:(n + 1),:]   # cls token 빼고 가져와 O
        x = self.encoder.dropout(x)
        
        
        glob = self.encoder.transformer(x)                  # transformer의 encoder 하나를 전체 통과한 output
                                                            # 아래는 transformer.encoder 중간 layer output features. hook=[2,5,8,11] 번째 에서 feat. 뽑기로 설정되어 O
        layer_1 = self.encoder.transformer.features[0]      # 중간 layer output feature from 2nd layer
        layer_2 = self.encoder.transformer.features[1]      # 중간 layer output feature from 5th layer
        layer_3 = self.encoder.transformer.features[2]      # 중간 layer output feature from 8th layer
        layer_4 = self.encoder.transformer.features[3]      # 중간 layer output feature from 11th layer     # 이게 마지막 layer outputd 이어서 glob랑 똑같네. interesting
        
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

        # breakpoint()
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

        pred_depth = self.scratch.output_conv(path_1) * self.max_depth  # self.max_depth=80 for kitti
        
        # pred_depth = (B,1,192,640)
        # features = len() = 4 = transformer 의 중간 layer outputs, (B,N,D)
        # fusion_features len() = 4 = 위 transformer 의 중간 layer outputs 를 CNN refinenet에 넣기 위해 img shape으로 여러 scale로 reshape한 (B,C,H,W) 꼴 
        
        outputs['pred_depth'] = pred_depth
        
        return outputs
    
    
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
        