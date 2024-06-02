import torch 
import torch.nn as nn 

import timm

class My_Decoder(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c    # 2048
        self.img_h = 192
        self.img_w = 640

        self.blk4 = nn.Sequential( nn.ConvTranspose2d(in_c, in_c//2, 3), nn.Upsample(size=(self.img_h//16, self.img_w//16), mode='bilinear', align_corners=True) )
        self.blk3 = nn.Sequential( nn.ConvTranspose2d(in_c, in_c//4, 3), nn.Upsample(size=(self.img_h//8, self.img_w//8), mode='bilinear', align_corners=True) )
        self.blk2 = nn.Sequential( nn.ConvTranspose2d(in_c//2, in_c//8, 3), nn.Upsample(size=(self.img_h//4, self.img_w//4), mode='bilinear', align_corners=True) )
        self.blk1 = nn.Sequential( nn.ConvTranspose2d(in_c//4, in_c//32, 3), nn.Upsample(size=(self.img_h//2, self.img_w//2), mode='bilinear', align_corners=True) )
        self.blk0 = nn.Sequential( nn.ConvTranspose2d(in_c//16, in_c//64, 3), nn.Upsample(size=(self.img_h, self.img_w), mode='bilinear', align_corners=True) )

    def forward(self, enc_out):
        '''    
        enc_out[4].shape (b, 2048, 6, 20)   h/32, w/32
        enc_out[3].shape (b, 1024, 12, 40)  h/16, w/16
        enc_out[2].shape (b, 512, 24, 80)   h/8, w/8
        enc_out[1].shape (b, 256, 48, 160)  h/4, w/4
        enc_out[0].shape (b, 64, 96, 320)   h/2, w/2
        '''
        
        dec_out3 = self.blk4(enc_out[4])    # (b,1024,12,40)
        dec_out2 = self.blk3( torch.cat([dec_out3, enc_out[3]], dim=1) )    # (b,512,24,80)
        dec_out1 = self.blk2( torch.cat([dec_out2, enc_out[2]], dim=1) )    # (b,256,48,160)
        dec_out0 = self.blk1( torch.cat([dec_out1, enc_out[1]], dim=1) )    # (b,64,96,320)
        dec_out =  self.blk0( torch.cat([dec_out0, enc_out[0]], dim=1) )    # (b,32,192,640)
        
        return dec_out


class My_Unet_Skip(nn.Module):
    def __init__(self, train_cfg):
        super().__init__() 
        
        self.enc_name = train_cfg.model.enc_name
        self.img_h = train_cfg.data.height
        self.img_w = train_cfg.data.width
        
        self.encoder=self.get_enc(enc_name=self.enc_name)
        self.decoder=self.get_dec(in_c=2048)
        self.head=self.get_head()
        
    def get_enc(self, enc_name):
        encoder = timm.create_model(enc_name, pretrained=True, features_only=True)
        return encoder
    
    def get_dec(self, in_c):
        decoder = My_Decoder(in_c=in_c)
        return decoder 
        
    def get_head(self):
        head = nn.Sequential( nn.ConvTranspose2d(self.decoder.in_c//64,1,3), nn.Upsample(size=(self.img_h, self.img_w), mode='bilinear', align_corners=True) )
        return head
        
    def forward(self, x):  
        enc_out = self.encoder(x)  
        dec_out = self.decoder(enc_out)
        pred_out = self.head(dec_out)
        return pred_out, None, None 