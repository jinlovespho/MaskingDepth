import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random

class SF_Depth_SelfSup_Try2(nn.Module):
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

        
        # Resnet
        self.encoder = encoder
        self.vit_features = vit_features 
        
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
        

        self.scratch = make_scratch(features, 256)
        self.scratch.refinenet1 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet2 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet3 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet4 = make_fusion_block(features=256, use_bn=False)

        # self.scratch.output_conv = head = nn.Sequential(
        #     nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
        #     Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        #     nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid(),
        #     nn.Identity(),
        # )
        
        self.conv_disp3= nn.Sequential( nn.Conv2d(256, 256//2, 3, 1, 1), 
                                        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                        nn.Conv2d(256//2, 32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid(),
                                        )
        
        self.conv_disp2= nn.Sequential( nn.Conv2d(256, 256//2, 3, 1, 1), 
                                        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                        nn.Conv2d(256//2, 32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid(),
                                        )
        
        self.conv_disp1= nn.Sequential( nn.Conv2d(256, 256//2, 3, 1, 1), 
                                        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                        nn.Conv2d(256//2, 32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid(),
                                        )
        
        self.conv_disp0= nn.Sequential( nn.Conv2d(256, 256//2, 3, 1, 1), 
                                        Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                                        nn.Conv2d(256//2, 32, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid(),
                                        )
        
        self.conv_tmp1=nn.Conv2d(256,768,3,1,1)
        self.conv_tmp2=nn.Conv2d(512,768,3,1,1)
        self.conv_tmp3=nn.Conv2d(1024,768,3,1,1)
        self.conv_tmp4=nn.Conv2d(2048,768,3,1,1)

    def forward(self, inputs, train_args, mode):
        outputs={}
        
        img_frames=[]
        if mode == 0:
            img_frames.append(inputs['color',  0, 0])   # curr_frame : [0] (b,3,192,640)
            # img_frames.append(inputs['color', -1, 0])   # prev_frame : [1]
            # img_frames.append(inputs['color',  1, 0])   # future_frame: [2]
        else:
            img_frames.append(inputs['color_aug',  0, 0])   
            # img_frames.append(inputs['color_aug', -1, 0])   
            # img_frames.append(inputs['color_aug',  1, 0])   
            
        
        tmp = self.encoder(img_frames[0])
        '''
        (Pdb) tmp[0].shape
        torch.Size([16, 64, 96, 320])
        (Pdb) tmp[1].shape
        torch.Size([16, 256, 48, 160])
        (Pdb) tmp[2].shape
        torch.Size([16, 512, 24, 80])
        (Pdb) tmp[3].shape
        torch.Size([16, 1024, 12, 40])
        (Pdb) tmp[4].shape
        torch.Size([16, 2048, 6, 20])'''
        
        tmp1 = nn.functional.interpolate(tmp[1], (12,40), mode='bilinear', align_corners=True)  # (b,256,12,40)
        tmp2 = nn.functional.interpolate(tmp[2], (12,40), mode='bilinear', align_corners=True)  # (b,512,12,40)
        tmp3 = nn.functional.interpolate(tmp[3], (12,40), mode='bilinear', align_corners=True)  # (b,1024,12,40)
        tmp4 = nn.functional.interpolate(tmp[4], (12,40), mode='bilinear', align_corners=True)  # (b,2048,6,20)
        
        layer_1 = self.conv_tmp1(tmp1)
        layer_2 = self.conv_tmp2(tmp2)
        layer_3 = self.conv_tmp3(tmp3)
        layer_4 = self.conv_tmp4(tmp4)

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
        path_4 = self.scratch.refinenet4(layer_4_rn)            # (b,256,12,40)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)    # (b,256,24,80)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)    # (b,256,48,160)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)    # (b,256,96,320)

        # pred_depth = self.scratch.output_conv(path_1) * self.max_depth  # (B,1,192,640)   self.max_depth=80 for kitti
        
        outputs['pred_disp',3] = self.conv_disp3(path_4)    # (b,1,12,40)   # passed through sigmoid. [0~1]
        outputs['pred_disp',2] = self.conv_disp2(path_3)    # (b,1,24,80)
        outputs['pred_disp',1] = self.conv_disp1(path_2)    # (b,1,48,160)
        outputs['pred_disp',0] = self.conv_disp0(path_1)    # (b,1,96,320)
        
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
        