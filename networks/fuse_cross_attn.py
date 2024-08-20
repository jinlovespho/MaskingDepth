import torch 
import torch.nn as nn

# concat cross_attn_map and cross_attn_output
class Fuse_Cross_Attn_Module1(nn.Module):
    def __init__(self, concat_dim, out_dim):
        super(Fuse_Cross_Attn_Module1, self).__init__()
        
        self.cat0_linear = nn.Linear(concat_dim, out_dim)
        self.cat1_linear = nn.Linear(concat_dim, out_dim)
        self.cat2_linear = nn.Linear(concat_dim, out_dim)
        self.cat3_linear = nn.Linear(concat_dim, out_dim)
        
    def forward(self, c_attn_maps, c_attn_outs):
        # c_attn_maps: (b,480,480)
        # c_attn_outs: (b,480,768)
        cat0 = torch.cat([c_attn_maps[0], c_attn_outs[0]], dim=2)   # (b,480,480+768)
        cat1 = torch.cat([c_attn_maps[1], c_attn_outs[1]], dim=2)
        cat2 = torch.cat([c_attn_maps[2], c_attn_outs[2]], dim=2)
        cat3 = torch.cat([c_attn_maps[3], c_attn_outs[3]], dim=2)
        
        cat0_linear = self.cat0_linear(cat0)                        # (b,480,768)
        cat1_linear = self.cat1_linear(cat1)
        cat2_linear = self.cat2_linear(cat2)
        cat3_linear = self.cat3_linear(cat3)
        
        return cat0_linear, cat1_linear, cat2_linear, cat3_linear 

# only add linear to cross_attn_map
class Fuse_Cross_Attn_Module2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Fuse_Cross_Attn_Module2, self).__init__()
        
        self.c_attn_map0_linear = nn.Linear(in_dim, out_dim)
        self.c_attn_map1_linear = nn.Linear(in_dim, out_dim)
        self.c_attn_map2_linear = nn.Linear(in_dim, out_dim)
        self.c_attn_map3_linear = nn.Linear(in_dim, out_dim)
        
        
    def forward(self, c_attn_maps, c_attn_outs):
        # c_attn_maps: (b,480,480)
        # c_attn_outs: (b,480,768)
        # breakpoint()
        out0_linear = self.c_attn_map0_linear(c_attn_maps[0])   # (b,480,480) -> (b,480,768)
        out1_linear = self.c_attn_map0_linear(c_attn_maps[1])
        out2_linear = self.c_attn_map0_linear(c_attn_maps[2])
        out3_linear = self.c_attn_map0_linear(c_attn_maps[3])
        
        # dot prod
        # out0 = out0_linear * c_attn_outs[0]
        # out1 = out1_linear * c_attn_outs[1]
        # out2 = out2_linear * c_attn_outs[2]
        # out3 = out3_linear * c_attn_outs[3]
        
        # sum
        out0 = out0_linear + c_attn_outs[0]     # (b,480,768)
        out1 = out1_linear + c_attn_outs[1]
        out2 = out2_linear + c_attn_outs[2]
        out3 = out3_linear + c_attn_outs[3]
        
        return out0, out1, out2, out3
    
# only add linear to cross_attn_outputs
class Fuse_Cross_Attn_Module3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Fuse_Cross_Attn_Module3, self).__init__()
        
        self.c_attn_out0_linear = nn.Linear(in_dim, out_dim)
        self.c_attn_out1_linear = nn.Linear(in_dim, out_dim)
        self.c_attn_out2_linear = nn.Linear(in_dim, out_dim)
        self.c_attn_out3_linear = nn.Linear(in_dim, out_dim)
        
        self.l0 = nn.Linear(out_dim, in_dim)
        self.l1 = nn.Linear(out_dim, in_dim)
        self.l2 = nn.Linear(out_dim, in_dim)
        self.l3 = nn.Linear(out_dim, in_dim)
        
        
    def forward(self, c_attn_maps, c_attn_outs):
        # c_attn_maps: (b,480,480)
        # c_attn_outs: (b,480,768)
        # breakpoint()
        out0_linear = self.c_attn_out0_linear(c_attn_outs[0])   # (b,480,480)
        out1_linear = self.c_attn_out0_linear(c_attn_outs[1])
        out2_linear = self.c_attn_out0_linear(c_attn_outs[2])
        out3_linear = self.c_attn_out0_linear(c_attn_outs[3])
        
        out0 = out0_linear + c_attn_maps[0]     # (b,480,480)
        out1 = out1_linear + c_attn_maps[1]
        out2 = out2_linear + c_attn_maps[2]
        out3 = out3_linear + c_attn_maps[3]
        
        out0 = self.l0(out0)    # (b,480,768)
        out1 = self.l0(out1)
        out2 = self.l0(out2)
        out3 = self.l0(out3)
        
        return out0, out1, out2, out3