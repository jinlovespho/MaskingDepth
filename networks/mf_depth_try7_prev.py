import torch
from torch import nn
from einops import repeat

from .vit import Transformer
from .dpt_utils import *
import random
import numpy as np
from einops import rearrange 

from .croco_blocks import *
from .fuse_cross_attn import *
from .conv4d import Conv4d_Module

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class MF_Depth_Try7(nn.Module):
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
        masking_ratio=0.5,
        cross_attn_depth = 8,
        croco = None,
    ):
        super().__init__()
    
        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
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
            nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True, dilation=1, groups=1), )
        
        self.act_postprocess2 = nn.Sequential(
            # readout_oper[1],
            Transpose(1, 2),
            nn.Conv2d(in_channels=vit_features,out_channels=features[1],kernel_size=1,stride=1,padding=0,),
            nn.ConvTranspose2d(in_channels=features[1],out_channels=features[1],kernel_size=2,stride=2,padding=0,bias=True,dilation=1,groups=1,),)
        
        self.act_postprocess3 = nn.Sequential(
            # readout_oper[2],
            Transpose(1, 2),
            nn.Conv2d(in_channels=vit_features,out_channels=features[2],kernel_size=1,stride=1,padding=0,),)
        
        self.act_postprocess4 = nn.Sequential(
            # readout_oper[3],
            Transpose(1, 2),
            nn.Conv2d(in_channels=vit_features,out_channels=features[3],kernel_size=1,stride=1,padding=0,),
            nn.Conv2d(in_channels=features[3],out_channels=features[3],kernel_size=3,stride=2,padding=1,),)
        
        self.unflatten = nn.Sequential(
            nn.Unflatten(2, torch.Size([self.num_pH, self.num_pW])))

        self.scratch = make_scratch(features, 256)
        self.scratch.refinenet1 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet2 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet3 = make_fusion_block(features=256, use_bn=False)
        self.scratch.refinenet4 = make_fusion_block(features=256, use_bn=False)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),  # scale factor 16 너무 크긴 해 
            nn.Conv2d(256 // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            nn.Identity(),
        ) 
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 256 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReLU(True),
            
            nn.Conv2d(256 // 2, 256//4, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReLU(True),
            
            nn.Conv2d(256 // 4, 256//8, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReLU(True),
            
            nn.Conv2d(256 // 8, 256//16, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.ReLU(True),
            
            nn.Conv2d(256//16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            nn.Identity(),     
        )

        
        # JINLOVESPHO
        self.masking_ratio=masking_ratio
        
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
        
        self.cross_attn_module4 = CrossAttention_Module(ca_dim=vit_features, ca_num_heads=self.encoder.heads, ca_depth=cross_attn_depth, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_mem=True, rope=self.rope)
        
        self.position_getter = PositionGetter()
        
        in_c=4*self.encoder.heads
        conv4d_out = 128
    
        # conv4d for depth
        self.conv4d_depth = nn.Sequential( Conv4d_Module(in_c=in_c,          out_c=conv4d_out//2, ks=(3,3,3,3), pd=(1,1,0,0), str=(1,1,1,1)),
                                           Conv4d_Module(in_c=conv4d_out//2, out_c=conv4d_out//2, ks=(3,3,3,3), pd=(1,1,0,0), str=(1,1,1,1)),
                                           Conv4d_Module(in_c=conv4d_out//2, out_c=conv4d_out//4, ks=(3,3,3,3), pd=(1,1,0,0), str=(1,1,1,1)),
                                           Conv4d_Module(in_c=conv4d_out//4, out_c=conv4d_out//4, ks=(3,3,3,3), pd=(1,1,0,0), str=(1,1,1,1)),
                                           Conv4d_Module(in_c=conv4d_out//4, out_c=conv4d_out//4, ks=(3,3,3,3), pd=(1,1,0,0), str=(1,1,1,1)),)
        # conv4d for pose
        self.conv4d_pose = nn.Sequential( Conv4d_Module(in_c=in_c,    out_c=in_c//2, ks=(3,3,3,3), pd=(1,1,1,1), str=(1,1,1,1)),
                                          Conv4d_Module(in_c=in_c//2, out_c=in_c//4, ks=(3,3,3,3), pd=(1,1,1,1), str=(1,1,1,1)),
                                          Conv4d_Module(in_c=in_c//4, out_c=in_c//4, ks=(3,3,3,3), pd=(1,1,1,1), str=(1,1,1,1)), )

        # linear for img recon.
        self.linear4_recon = nn.Linear(vit_features, 3*self.pH*self.pW)
        
        # linear for depth pred.
        self.linear4_depth = nn.Linear(1920, 256)
        
    def forward(self, inputs, train_args, mode):
        
        outputs={}
        
        img_frames=[]
        if mode == 0:
            img_frames.append(inputs['color',  0, 0])   # curr_frame : [0]
            img_frames.append(inputs['color', -1, 0])   # prev_frame : [1]
            # img_frames.append(inputs['color',  1, 0])   # future_frame: [2]
        else:
            img_frames.append(inputs['color_aug',  0, 0])   
            img_frames.append(inputs['color_aug', -1, 0])   
            # img_frames.append(inputs['color_aug',  1, 0])   


        # tokenize input image frames(t,t-1, . . ) and add positional embeddings
        tokenized_frames = []
        poses = []
        for i, frame in enumerate(img_frames):              # frame: (b,3,192,640)   
            tmp =self.encoder.to_patch_embedding(frame)     # tmp: (b,480,768), where patchsize=16, (192/16)*(640/16)=480 총 token(patch)개수 
            if not self.encoder.croco:                      
                tmp += self.encoder.pos_emb_lst[i][:,1:,:]
            tmp = self.encoder.dropout(tmp)
            tokenized_frames.append(tmp)

            # poses.append(self.position_getter(frame.shape[0],frame.shape[2]//16,frame.shape[3]//16, frame.device))
            poses.append(self.position_getter(frame.shape[0],frame.shape[2]//16,frame.shape[3]//16))
        
        # batch_size, length, dim, device
        b,n,dim = tokenized_frames[0].shape     # (8,480,768)
        
        # number of patches to mask
        num_p_msk = int(self.masking_ratio * n ) if mode == 0 else 0    # validation(mode=1) 이면 num_p_mask=0 으로 만들어서 unmask !
        
        # random masking index generation
        idx_rnd = torch.rand(b,n).argsort()
        idx_msk, idx_umsk = idx_rnd[:,:num_p_msk], idx_rnd[:,num_p_msk:]
        idx_msk = idx_msk.sort().values
        idx_umsk = idx_umsk.sort().values
        idx_bs = torch.arange(b)[:,None]
        
        # unmasked tokens
        frame0_unmsk_tkn = tokenized_frames[0][idx_bs, idx_umsk]
        poses0_unmsk_tkn = poses[0][idx_bs, idx_umsk]
        
        # manually put cpu variables to gpu 
        poses0_unmsk_tkn = poses0_unmsk_tkn.to(self.device)
        poses[0] = poses[0].to(self.device)
        poses[1] = poses[1].to(self.device)

        # masked tokens
        msk_tkns1 = repeat(self.msk_tkn1, 'd -> b n d', b=b, n=num_p_msk)
        msk_tkns2 = repeat(self.msk_tkn2, 'd -> b n d', b=b, n=num_p_msk) 
        msk_tkns3 = repeat(self.msk_tkn3, 'd -> b n d', b=b, n=num_p_msk) 
        msk_tkns4 = repeat(self.msk_tkn4, 'd -> b n d', b=b, n=num_p_msk) 
        
        # t frame encoder
        glob1 = self.encoder.transformer(frame0_unmsk_tkn, poses0_unmsk_tkn)     # (8,192,768)
        glob1_layer_1 = self.encoder.transformer.features[0]                    # (8,192,768)
        glob1_layer_2 = self.encoder.transformer.features[1]      
        glob1_layer_3 = self.encoder.transformer.features[2]      
        glob1_layer_4 = self.encoder.transformer.features[3]    # glob1 과 동일          
        
        # t-1 frame encoder
        glob2 = self.encoder.transformer(tokenized_frames[1],poses[1])          # (8,480,768)
        glob2_layer_1 = self.encoder.transformer.features[0]                    # (8,480,768)
        glob2_layer_2 = self.encoder.transformer.features[1]      
        glob2_layer_3 = self.encoder.transformer.features[2]      
        glob2_layer_4 = self.encoder.transformer.features[3]      

        # encoder output1 에 msk tkn 을 concat 하는 과정
        frame0_msk_umsk_tkn1 = torch.zeros(b, n, dim, device=self.device)            # (8,480,768)
        frame0_msk_umsk_tkn1[idx_bs, idx_umsk] = glob1_layer_1
        frame0_msk_umsk_tkn1[idx_bs, idx_msk] = msk_tkns1

        if not self.encoder.croco:
            frame0_msk_umsk_tkn1 = frame0_msk_umsk_tkn1 + self.decoder_pose_embed
            glob2_layer_1 = glob2_layer_1 + self.decoder_pose_embed
        
        # encoder output2 에 msk tkn 을 concat 하는 과정
        frame0_msk_umsk_tkn2 = torch.zeros(b, n, dim, device=self.device)            # (8,480,768)
        frame0_msk_umsk_tkn2[idx_bs, idx_umsk] = glob1_layer_2
        frame0_msk_umsk_tkn2[idx_bs, idx_msk] = msk_tkns2

        if not self.encoder.croco:
            frame0_msk_umsk_tkn2 = frame0_msk_umsk_tkn2 + self.decoder_pose_embed
            glob2_layer_2 = glob2_layer_2 + self.decoder_pose_embed
        
        # encoder output3 에 msk tkn 을 concat 하는 과정
        frame0_msk_umsk_tkn3 = torch.zeros(b, n, dim, device=self.device)            # (8,480,768)
        frame0_msk_umsk_tkn3[idx_bs, idx_umsk] = glob1_layer_3 
        frame0_msk_umsk_tkn3[idx_bs, idx_msk] = msk_tkns3

        if not self.encoder.croco:
            frame0_msk_umsk_tkn3 = frame0_msk_umsk_tkn3 + self.decoder_pose_embed
            glob2_layer_3 = glob2_layer_3 + self.decoder_pose_embed

        # encoder output4 에 msk tkn 을 concat 하는 과정
        frame0_msk_umsk_tkn4 = torch.zeros(b, n, dim, device=self.device)            # (8,480,768)
        frame0_msk_umsk_tkn4[idx_bs, idx_umsk] = glob1_layer_4
        frame0_msk_umsk_tkn4[idx_bs, idx_msk] = msk_tkns4

        if not self.encoder.croco:
            frame0_msk_umsk_tkn4 = frame0_msk_umsk_tkn4 + self.decoder_pose_embed
            glob2_layer_4 = glob2_layer_4 + self.decoder_pose_embed
            
        # cross attention between t features and t-1 features
        # and obtain ca_out and ca_map
        ca_out4, ca_map4 = self.cross_attn_module4(frame0_msk_umsk_tkn4, glob2_layer_4, poses[0], poses[1])

        cat_ca_maps4 = torch.cat(ca_map4, dim=1)    # (b,4*head,n,n)    #(b,48,480,480)
        cat_ca_maps4 = cat_ca_maps4.view(b, 4*self.encoder.heads, self.num_pH, self.num_pW, self.num_pH, self.num_pW )  # (b,head,h,w,h,w)  8,12,12,40,12,40)
        
        # conv4d 
        ca_map4_depth = self.conv4d_depth(cat_ca_maps4)   # (b, out_c, h1,w1,h2,w2)    (b,128, 12,40,12,40)
        ca_map4_pose = self.conv4d_pose(cat_ca_maps4)     # (b, 12, 12,40,12,40)
        

        # pred depth
        ca_map4_depth = rearrange(ca_map4_depth, 'b c h1 w1 h2 w2 -> b (h1 w1) (c h2 w2)')
        pred_depth4 = self.linear4_depth(ca_map4_depth)
        pred_depth4 = rearrange(pred_depth4, 'b (h w) c -> b c h w', h=12, w=40)    # (b,256,12,40)
        pred_depth4 = self.depth_head(pred_depth4) # (b,1,192,640)
        
        # recon img
        recon_img4 = self.linear4_recon(ca_out4)    # (b,480,768)
        recon_img4 = rearrange(recon_img4, 'b (num_pH num_pW) (c pH pW) -> b c (num_pH pH) (num_pW pW)', num_pH=self.num_pH, num_pW=self.num_pW, pH=self.pH, pW=self.pW)   # (b,3,192,640)
    
        
        outputs['pred_depth4'] = pred_depth4
        outputs['recon_img4'] = recon_img4
        outputs['ca_map4_pose'] = ca_map4_pose
        
        return outputs
        
        breakpoint()
        
        axis_trans = self.conv4d_pose_decoder(cat_ca_maps1)    # (b,12, 12,40,12,40)
        axis_trans = axis_trans.mean(dim=[2,3,4,5]).view(-1,2,1,6) # (b,12) -> (b,2,1,6)
        axisangle = axis_trans[..., :3]     # (b,2,1,3)
        translation = axis_trans[..., 3:]   # (b,2,1,3)
        rel_pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)  # (b,4,4)
        
        
        bs, out_c, h1, w1, h2, w2 = cmap4.shape
        cmap4 = cmap4.view(bs, out_c, h1*w1, h2*w2).permute(0,2,1,3).flatten(start_dim=2)   # (b,n, out_c*h2*w2) (b,480,26112)
        
        lin_cmap4 = self.cmap4_linear(cmap4)
        
        layer_4 = ca_out4 + lin_cmap4
        
        
    
    
    def rollout(self, attentions, discard_ratio, head_fusion, device):
        # breakpoint()
        result = torch.eye(attentions[0].size(-1)).to(device)
        with torch.no_grad():
            for attention in attentions:
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

# class PositionGetter(object):
#     """ return positions of patches """

#     def __init__(self):
#         self.cache_positions = {}
        
#     def __call__(self, b, h, w, device):
#         # ForkedPdb().set_trace()
        
#         if not (h,w) in self.cache_positions:
#             x = torch.arange(w, device=device)
#             y = torch.arange(h, device=device)
#             self.cache_positions[h,w] = torch.cartesian_prod(y, x) # (h, w, 2)
#         pos = self.cache_positions[h,w].view(1, h*w, 2).expand(b, -1, 2).clone()
#         return pos
    
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}
        
    def __call__(self, b, h, w):
        # ForkedPdb().set_trace()
        
        if not (h,w) in self.cache_positions:
            x = torch.arange(w)
            y = torch.arange(h)
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