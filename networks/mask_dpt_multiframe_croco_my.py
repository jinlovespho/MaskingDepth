import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random
import numpy as np


from .croco_blocks import *
from .fuse_cross_attn import *


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
        cross_attn_depth = 8,
        croco = None
    ):
        super().__init__()
        
        # JINLOVESPHO
        self.num_prev_frame=num_prev_frame
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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

        if croco is not None:
            self.rope = RoPE2D(freq=100)
        else:
            self.rope = None
        # self.mask_pe_table = nn.Embedding(encoder.num_patches, vit_features)
        self.decoder_pose_embed = nn.Parameter(torch.from_numpy(get_2d_sincos_pos_embed(vit_features, self.target_size[0]//16,self.target_size[1]//16, 0)).float(), requires_grad=False)
        
        self.cross_attn_module1 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=self.rope)
        self.cross_attn_module2 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=self.rope)
        self.cross_attn_module3 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=self.rope)
        self.cross_attn_module4 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=self.rope)
        
        self.position_getter = PositionGetter()
        
        # self.c_attn_map1 = []
        # self.c_attn_map2 = []
        # self.c_attn_map3 = []
        # self.c_attn_map4 = []
        
        # # set hooks to obtain cross attention maps
        # for name, module in self.cross_attn_module1.named_modules():
        #     if 'cross_attn.attn_drop' in name:    
        #         # c_attn_map1.append(name)
        #         module.register_forward_hook(lambda m,i,o: self.c_attn_map1.append(o) )    
                   
        # for name, module in self.cross_attn_module2.named_modules():
        #     if 'cross_attn.attn_drop' in name:    
        #         # c_attn_map2.append(name)
        #         module.register_forward_hook(lambda m,i,o: self.c_attn_map2.append(o) ) 
                
        # for name, module in self.cross_attn_module3.named_modules():
        #     if 'cross_attn.attn_drop' in name:    
        #         # c_attn_map3.append(name)   
        #         module.register_forward_hook(lambda m,i,o: self.c_attn_map3.append(o) ) 
                
        # for name, module in self.cross_attn_module4.named_modules():
        #     if 'cross_attn.attn_drop' in name:    
        #         # c_attn_map4.append(name)
        #         module.register_forward_hook(lambda m,i,o: self.c_attn_map4.append(o) ) 
                
        # JINLOVESPHO
        self.fuse_cross_attn_module1 = Fuse_Cross_Attn_Module1(concat_dim=480+768, out_dim=768)
        # self.fuse_cross_attn_module2 = Fuse_Cross_Attn_Module2(in_dim=480, out_dim=768)
        # self.fuse_cross_attn_module3 = Fuse_Cross_Attn_Module3(in_dim=768, out_dim=480)
        
    
    def forward(self, img_frames, K = 1, mode=None):
        # breakpoint()
        # tokenize input image frames(t,t-1, . . ) and add positional embeddings
        tokenized_frames = []
        poses = []
        for i, frame in enumerate(img_frames):              # frame: (b,3,192,640)   
            tmp =self.encoder.to_patch_embedding(frame)     # embedding: (b,480,768), where patchsize=16, 480 = 12*40 = (192/16)*(640/16) 총 token(patch)개수 
            if not self.encoder.croco:                      
                tmp += self.encoder.pos_emb_lst[i][:,1:,:]
            tmp = self.encoder.dropout(tmp)
            tokenized_frames.append(tmp)

            poses.append(self.position_getter(frame.shape[0],frame.shape[2]//16,frame.shape[3]//16, frame.device))
        
        # batch_size, length, dim, device
        b,n,dim = tokenized_frames[0].shape     # (8,480,768)
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
        poses0_unmsk_tkn = poses[0][idx_bs, idx_umsk]

        # masked tokens
        msk_tkns1 = repeat(self.msk_tkn1, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns2 = repeat(self.msk_tkn2, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns3 = repeat(self.msk_tkn3, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns4 = repeat(self.msk_tkn4, 'd -> b n d', b=b, n=num_p_msk)

        # pos_msk_tkns = self.mask_pe_table(idx_msk)
        # pos_umsk_tkns = self.mask_pe_table(idx_umsk)
        
        # add positional embedding for masked tokens
        
        # t frame encoder
        glob1 = self.encoder.transformer(frame0_unmsk_tkn,poses0_unmsk_tkn)     # (8,192,768)
        glob1_layer_1 = self.encoder.transformer.features[0]                    # (8,192,768)
        glob1_layer_2 = self.encoder.transformer.features[1]      
        glob1_layer_3 = self.encoder.transformer.features[2]      
        glob1_layer_4 = self.encoder.transformer.features[3]      
                    
        # t-1 frame encoder
        glob2 = self.encoder.transformer(tokenized_frames[1],poses[1])          # (8,480,768)
        glob2_layer_1 = self.encoder.transformer.features[0]                    # (8,480,768)
        glob2_layer_2 = self.encoder.transformer.features[1]      
        glob2_layer_3 = self.encoder.transformer.features[2]      
        glob2_layer_4 = self.encoder.transformer.features[3]      

        frame0_msk_umsk_tkn1 = torch.zeros(b, n, dim, device=device)            # (8,480,768)
        frame0_msk_umsk_tkn1[idx_bs, idx_umsk] = glob1_layer_1
        frame0_msk_umsk_tkn1[idx_bs, idx_msk] = msk_tkns1

        if not self.encoder.croco:
            frame0_msk_umsk_tkn1 = frame0_msk_umsk_tkn1 + self.decoder_pose_embed
            glob2_layer_1 = glob2_layer_1 + self.decoder_pose_embed
        
        frame0_msk_umsk_tkn2 = torch.zeros(b, n, dim, device=device)            # (8,480,768)
        frame0_msk_umsk_tkn2[idx_bs, idx_umsk] = glob1_layer_2
        frame0_msk_umsk_tkn2[idx_bs, idx_msk] = msk_tkns2

        if not self.encoder.croco:
            frame0_msk_umsk_tkn2 = frame0_msk_umsk_tkn2 + self.decoder_pose_embed
            glob2_layer_2 = glob2_layer_2 + self.decoder_pose_embed
        
        frame0_msk_umsk_tkn3 = torch.zeros(b, n, dim, device=device)            # (8,480,768)
        frame0_msk_umsk_tkn3[idx_bs, idx_umsk] = glob1_layer_3 
        frame0_msk_umsk_tkn3[idx_bs, idx_msk] = msk_tkns3

        if not self.encoder.croco:
            frame0_msk_umsk_tkn3 = frame0_msk_umsk_tkn3 + self.decoder_pose_embed
            glob2_layer_3 = glob2_layer_3 + self.decoder_pose_embed

        frame0_msk_umsk_tkn4 = torch.zeros(b, n, dim, device=device)            # (8,480,768)
        frame0_msk_umsk_tkn4[idx_bs, idx_umsk] = glob1_layer_4
        frame0_msk_umsk_tkn4[idx_bs, idx_msk] = msk_tkns4

        if not self.encoder.croco:
            frame0_msk_umsk_tkn4 = frame0_msk_umsk_tkn4 + self.decoder_pose_embed
            glob2_layer_4 = glob2_layer_4 + self.decoder_pose_embed
           
        # JINLOVESPHO
        c_attn_map1 = []
        c_attn_map2 = []
        c_attn_map3 = []
        c_attn_map4 = []
        
        # set hooks to obtain cross attention maps
        for name, module in self.cross_attn_module1.named_modules():
            if 'cross_attn.attn_drop' in name:    
                # tmp.append(name)
                module.register_forward_hook(lambda m,i,o: c_attn_map1.append(o.detach().cpu()) )    
                   
        for name, module in self.cross_attn_module2.named_modules():
            if 'cross_attn.attn_drop' in name:    
                # c_attn_map2.append(name)
                module.register_forward_hook(lambda m,i,o: c_attn_map2.append(o.detach().cpu()) ) 
                
        for name, module in self.cross_attn_module3.named_modules():
            if 'cross_attn.attn_drop' in name:    
                # c_attn_map3.append(name)   
                module.register_forward_hook(lambda m,i,o: c_attn_map3.append(o.detach().cpu()) ) 
                
        for name, module in self.cross_attn_module4.named_modules():
            if 'cross_attn.attn_drop' in name:    
                # c_attn_map4.append(name)
                module.register_forward_hook(lambda m,i,o: c_attn_map4.append(o.detach().cpu()) ) 
            
        # breakpoint()
        # cross attention output
        cross_attn_out1, _ = self.cross_attn_module1(frame0_msk_umsk_tkn1, glob2_layer_1, poses[0], poses[1])      # (b,480,768)
        cross_attn_out2, _ = self.cross_attn_module2(frame0_msk_umsk_tkn2, glob2_layer_2, poses[0], poses[1])
        cross_attn_out3, _ = self.cross_attn_module3(frame0_msk_umsk_tkn3, glob2_layer_3, poses[0], poses[1])
        cross_attn_out4, _ = self.cross_attn_module4(frame0_msk_umsk_tkn4, glob2_layer_4, poses[0], poses[1])
        
        # Attention Rollout
        discard_ratio=0.0
        head_fusion='mean'
        # c_attn_map1_rollout = self.rollout(c_attn_map1, discard_ratio, head_fusion, self.device)    # (b,480,480)
        # c_attn_map2_rollout = self.rollout(c_attn_map2, discard_ratio, head_fusion, self.device)
        # c_attn_map3_rollout = self.rollout(c_attn_map3, discard_ratio, head_fusion, self.device)
        # c_attn_map4_rollout = self.rollout(c_attn_map4, discard_ratio, head_fusion, self.device)
        
        # # breakpoint()
        # c_attn_map1_rollout = c_attn_map1_rollout.detach()      # (b,480,480)
        # c_attn_map2_rollout = c_attn_map2_rollout.detach()
        # c_attn_map3_rollout = c_attn_map3_rollout.detach()
        # c_attn_map4_rollout = c_attn_map4_rollout.detach()
        
        # breakpoint()
        # WAY1 - concat 
        # layer_1, layer_2, layer_3, layer_4 = self.fuse_cross_attn_module1( c_attn_maps = [c_attn_map1_rollout, c_attn_map2_rollout, c_attn_map3_rollout, c_attn_map4_rollout],
        #                                                                    c_attn_outs = [cross_attn_out1, cross_attn_out2, cross_attn_out3, cross_attn_out4])
        
        # # WAY2 - only c_attn_map linear
        # layer_1, layer_2, layer_3, layer_4 = self.fuse_cross_attn_module2( c_attn_maps = [c_attn_map1_rollout, c_attn_map2_rollout, c_attn_map3_rollout, c_attn_map4_rollout],
        #                                                                    c_attn_outs = [cross_attn_out1, cross_attn_out2, cross_attn_out3, cross_attn_out4])
        
        # # WAY3 - onlt c_attn_out linear 
        # layer_1, layer_2, layer_3, layer_4 = self.fuse_cross_attn_module3( c_attn_maps = [c_attn_map1_rollout, c_attn_map2_rollout, c_attn_map3_rollout, c_attn_map4_rollout],
        #                                                                    c_attn_outs = [cross_attn_out1, cross_attn_out2, cross_attn_out3, cross_attn_out4])
    
    
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
    
    
    def rollout(self, attentions, discard_ratio, head_fusion, device):
        # breakpoint()
        result = torch.eye(attentions[0].size(-1)).to(device)
        with torch.no_grad():
            for attention in attentions:
                attention = attention.to(device)
                if head_fusion == "mean":                                   # attention (b,3,197,197)
                    attention_heads_fused = attention.mean(axis=1)          # attention_heads_fuse (b,197,197)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"

                # Drop the lowest attentions, but
                # don't drop the class token``
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)    
                num_to_discard = int(flat.size(-1)*discard_ratio)
                _, indices = flat.topk(num_to_discard, -1, False) 
                indices = indices[indices != 0]    
                flat[0, indices] = 0   
                 
                I = torch.eye(attention_heads_fused.size(-1)).to(device)    
                a = (attention_heads_fused + 1.0*I)/2               
                a = a / a.sum(dim=-1, keepdim=True)   
                
                result = torch.matmul(a, result)      
                 
                return result
    
    
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
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
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

class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w, device):
        if not (h,w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
        pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
        return pos
    

try:
    from networks.curope import cuRoPE2D
    RoPE2D = cuRoPE2D
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')

    class RoPE2D(torch.nn.Module):
        
        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq 
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D,seq_len,device,dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos() # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D,seq_len,device,dtype] = (cos,sin)
            return self.cache[D,seq_len,device,dtype]
            
        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim==2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)
            
        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3)%2==0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim==3 and positions.shape[-1] == 2 # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max())+1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:,:,0], cos, sin)
            x = self.apply_rope1d(x, positions[:,:,1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens