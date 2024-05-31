import torch 
import torch.nn as nn 

import timm

class My_Decoder(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.blk1 = nn.Sequential( nn.ConvTranspose2d(in_c,         int(in_c/2), 3), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) )
        self.blk2 = nn.Sequential( nn.ConvTranspose2d(int(in_c/2), int(in_c/4), 3), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) )
        self.blk3 = nn.Sequential( nn.ConvTranspose2d(int(in_c/4), int(in_c/8), 3), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) )
        self.blk4 = nn.Sequential( nn.ConvTranspose2d(int(in_c/8), int(in_c/16), 3), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) )
        self.blk5 = nn.Sequential( nn.ConvTranspose2d(int(in_c/16), int(in_c/32), 3), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) )

    def forward(self, x):
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)

        return x 


class My_Unet(nn.Module):
    def __init__(self, enc_name):
        super().__init__() 

        self.encoder=self.get_enc(enc_name=enc_name)
        self.decoder=self.get_dec(in_c=2048)
        self.head=self.get_head()
        
    def get_enc(self, enc_name):
        encoder = timm.create_model(enc_name, pretrained=False, features_only=True)
        return encoder
    
    def get_dec(self, in_c):
        decoder = My_Decoder(in_c=in_c)
        return decoder 
        
    def get_head(self):
        head = nn.Sequential( nn.ConvTranspose2d(64,1,3), nn.Upsample(size=(192,640), mode='bilinear', align_corners=True) )
        return head
        
    def forward(self, x):  
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out[-1])
        pred_out = self.head(dec_out)
        return pred_out, None, None 