import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random

from .croco_blocks import *

class Masked_DPT_Multiframe_CrossAttn_mask_t(nn.Module):
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
        num_prev_frame=None,
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
        
        # mask tokens for each frame 
        msk_tkn_lst = []
        for _ in range(self.num_prev_frame+1):
            tmp = nn.Parameter(torch.randn(vit_features))
            msk_tkn_lst.append(tmp)

        self.msk_tkn_lst = nn.ParameterList(msk_tkn_lst)
        
        self.cross_attn_module = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=None)
        
    

    def forward(self, img_frames, K = 1, mode=None):
        # assert mask_ratio >= 0 and mask_ratio < 1, 'masking ratio must be kept between 0 and 1' 
        
        # img_frames 는 리스트 꼴로 [ current_frame, prev_frame, prev2_frame ] 등의 순서로 담겨있다.
        # 하나의 frame shape=(B, 3, 192, 640)
          
        # patchify each frames in img_frames
        tokenized_frames = []
        for i, frame in enumerate(img_frames):
            tmp=self.encoder.to_patch_embedding(frame)
            tmp += self.encoder.pos_emb_lst[i][:,1:,:]
            tmp = self.encoder.dropout(tmp)
            tokenized_frames.append(tmp)
        
        b,n,_ = tokenized_frames[0].shape       
        # breakpoint()
        # if mode == TRAIN(0) add mask tokens to the input
        if mode==0 :         
            # set num_frame_to_msk = 1 to only mask the t frame
            # set num_frame_to_msk = self.num_prev_frame+1 to mask all frames(t,t-1,t-2, . . .)
            num_frame_to_msk = self.num_frame_to_mask
            num_p_msk = int(self.masking_ratio * n )
            
            for i in range(num_frame_to_msk):
                # expand mask_token (1,1,D) to (B,N,D) shape and add positional embedding
                msk_tkn = repeat(self.msk_tkn_lst[i], 'd -> b n d', b=b, n=n )
                msk_tkn.data = msk_tkn.clone().detach() + self.encoder.pos_emb_lst[i].data[:,1:,:]
                
                # calculate of patches needed to be masked, and get positions (indices) to be masked
                msk_idx = torch.rand(b, n, device='cuda').topk(k=num_p_msk, dim=-1).indices
                bool_msk = torch.zeros(b,n,device='cuda').scatter_(dim=-1, index=msk_idx, value=1).bool()
                tokenized_frames[i] = torch.where(bool_msk[...,None], msk_tkn, tokenized_frames[i])
        
        # concated_frames_tensor = torch.concat(tokenized_frames, dim=1)       # (B, 960,768) cuz 480*2 개를 concat 했기에
        
        if K == 1:
            
            glob2 = self.encoder.transformer(tokenized_frames[1])
            glob1 = self.encoder.transformer(tokenized_frames[0])   # t-th frame을 마지막에 통과시켜야 아래에 layer_1~layer_4 정보가 t-th frame정보로 채워진다 !!!
             
            cross_attn_out = self.cross_attn_module(glob1, glob2) 
            
            # glob = self.encoder.transformer(concated_frames_tensor)     # 이렇게 한번 encoder.transformer을 통과 시켜야 내부 self.features에 중간 layer output 결과 담긴다
                                                                # 아래는 transformer.encoder 중간 layer output features. hook=[2,5,8,11] 번째 에서 feat. 뽑기로 설정되어 O
            
            layer_1 = self.encoder.transformer.features[0]      # 중간 layer output feature from 2nd layer
            layer_2 = self.encoder.transformer.features[1]      # 중간 layer output feature from 5th layer
            layer_3 = self.encoder.transformer.features[2]      # 중간 layer output feature from 8th layer
            layer_4 = self.encoder.transformer.features[3]      # 중간 layer output feature from 11th layer     # 이게 마지막 layer outputd 이어서 glob랑 똑같네. interesting
            
            layer_4 = layer_4 + cross_attn_out
            
            # breakpoint()
            
            # self.act_postprocess1 구성 => Transpose(), Conv2d(), ConvTranspose2d()
            # self.act_postprocess2 구성 => Transpose(), Conv2d(), ConvTranpose2d()
            # self.act_postprocess3 구성 => Tranpose(), Conv2d()
            # self.act_postprocess4 구성 => Tranpose(), Conv2d(), Conv2d()
            # 이렇게 하는 이유: transformer 의 output features (B,480,768) 을 이제 "CNN" decoder에 넣어야하기에 다시 img 꼴로 reshape 해줘야 O
            # CNN 이 받는 input img shape은 (C,H,W) 이기에, embed_dim은 C해당하므로, 768을 앞으로 가져오고 480을 뒤로 보내는 것. 이해 O
            
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

        # breakpoint()

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
        