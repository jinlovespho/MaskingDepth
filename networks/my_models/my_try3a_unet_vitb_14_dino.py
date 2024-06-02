import torch 
import torch.nn as nn 

import timm

class My_Decoder(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c    # 768
        self.img_h = 196
        self.img_w = 644
        self.num_patch_h = 196//14  # 14
        self.num_patch_w = 644//14  # 46

        self.blk4 = nn.Sequential( nn.ConvTranspose2d(in_c, in_c//2, 3), nn.Upsample(size=(self.img_h//14, self.img_w//14), mode='bilinear', align_corners=True) )
        self.blk3 = nn.Sequential( nn.ConvTranspose2d(in_c//2, in_c//4, 3), nn.Upsample(size=(self.img_h//7, self.img_w//7), mode='bilinear', align_corners=True) )
        self.blk2 = nn.Sequential( nn.ConvTranspose2d(in_c//4, in_c//8, 3), nn.Upsample(size=(self.img_h//4, self.img_w//4), mode='bilinear', align_corners=True) )
        self.blk1 = nn.Sequential( nn.ConvTranspose2d(in_c//8, in_c//16, 3), nn.Upsample(size=(self.img_h//2, self.img_w//2), mode='bilinear', align_corners=True) )
        self.blk0 = nn.Sequential( nn.ConvTranspose2d(in_c//16, in_c//32, 3), nn.Upsample(size=(self.img_h, self.img_w), mode='bilinear', align_corners=True) )

    def forward(self, enc_out):
        '''    
        enc_out (b,644,768) 644 = 14*46
        
        '''

        b,n,d = enc_out.shape
        reshaped_enc_out = enc_out.permute(0,2,1).view(b,d, self.num_patch_h, self.num_patch_w) # (b,768,14,46)
        
        dec_out3 = self.blk4( reshaped_enc_out )    # (b,384,14,46)
        dec_out2 = self.blk3( dec_out3 )    # (b,192,28,192)
        dec_out1 = self.blk2( dec_out2 )    # (b,96,49,161)
        dec_out0 = self.blk1( dec_out1 )    # (b,48,98,322)
        dec_out =  self.blk0( dec_out0 )    # (b,24,196,644)
        
        return dec_out


class My_Unet_Enc_VitB_Dinov2(nn.Module):
    def __init__(self, train_cfg):
        super().__init__() 
        
        self.enc_name = train_cfg.model.enc_name
        self.enc_is_pretrained = train_cfg.model.enc_is_pretrained
        self.img_h = train_cfg.data.height
        self.img_w = train_cfg.data.width
        
        self.encoder=self.get_enc(enc_name=self.enc_name)
        self.decoder=self.get_dec(in_c=train_cfg.model.enc_hidden_dim)
        self.head=self.get_head()
        
    def get_enc(self, enc_name):
        encoder = timm.create_model(enc_name, pretrained=self.enc_is_pretrained, img_size=(self.img_h,self.img_w))
        return encoder
    
    def get_dec(self, in_c):
        decoder = My_Decoder(in_c=in_c)
        return decoder 
        
    def get_head(self):
        head = nn.Sequential( nn.ConvTranspose2d(self.decoder.in_c//32,1,3), nn.Upsample(size=(self.img_h, self.img_w), mode='bilinear', align_corners=True) )
        return head
        
    def forward(self, x):  
        enc_out = self.encoder.forward_features(x)  
        enc_out = enc_out[:,1:,:]   # remove cls tkn
        dec_out = self.decoder(enc_out)
        pred_out = self.head(dec_out)
        return pred_out, None, None 