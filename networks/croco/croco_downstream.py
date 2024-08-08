# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# CroCo model for downstream tasks
# --------------------------------------------------------

import torch
import torch.nn as nn
from einops import rearrange, repeat
import random

from .croco import CroCoNet
from .blocks import Conv4d_Module



def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt: # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'): # pretrained using the official code release
        s = ckpt['args'].model # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict'+s[len('CroCoNet'):]) # transform it into the string of a dictionary and evaluate it
    else: # CroCo v1 released models
        return dict()

class CroCoDownstreamMonocularEncoder(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for monocular downstream task, only using the encoder.
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        NOTE: It works by *calling super().__init__() but with redefined setters
        
        """
        super(CroCoDownstreamMonocularEncoder, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_decoder(self, *args, **kwargs):
        """ No decoder """
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No 'prediction head' for downstream tasks."""
        return

    def forward(self, img):
        """
        img if of size batch_size x 3 x h x w
        """
        B, C, H, W = img.size()
        img_info = {'height': H, 'width': W}
        need_all_layers = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, _, _ = self._encode_image(img, do_mask=False, return_all_blocks=need_all_layers)
        return self.head(out, img_info)
        
        
class CroCoDownstreamBinocular(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(CroCoDownstreamBinocular, self).__init__(**kwargs)
        head.setup(self)
        self.head = head
        self.attn_conv4d = kwargs.get('attn_conv4d', False)

    # def _set_mask_generator(self, *args, **kwargs):
    #     """ No mask generator """
    #     return

    # def _set_mask_token(self, *args, **kwargs):
    #     """ No mask token """
    #     self.mask_token = None
    #     return

    # def _set_prediction_head(self, *args, **kwargs):
    #     """ No prediction head for downstream tasks, define your own head """
    #     return
        
    def encode_image_pairs(self, img1, img2, return_all_blocks=False,mode=0):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        do_mask = (mode==0) #and random.random() > 0.5

        out, pos, mask1 = self._encode_image(img1, do_mask=do_mask, return_all_blocks=return_all_blocks)
        out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version
        # out, pos, _ = self._encode_image( torch.cat( (img1,img2), dim=0), do_mask=False, return_all_blocks=return_all_blocks )
        # if return_all_blocks:
        #     out,out2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in out])))
        #     out2 = out2[-1]
        # else:
        #     out,out2 = out.chunk(2, dim=0)
        # pos,pos2 = pos.chunk(2, dim=0)            
        return out, out2, pos, pos2, mask1

    def forward(self, img1, img2, mode, intrinsics=None):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks
        out, out2, pos, pos2, mask1 = self.encode_image_pairs(img1, img2, return_all_blocks=return_all_blocks, mode=mode)
        
        if self.args.encoder_freeze:
            out = [o.detach() for o in out]
            out2 = out2.detach()
        
        if return_all_blocks:
            decout,attn_map, f1 = self._decoder(out[-1], pos, mask1, out2, pos2, return_all_blocks=return_all_blocks)
            decout = out+decout
            # decout = [d.detach() for d in decout]
        else:
            decout,attn_map,f1 = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)#.detach()
            
            
            
        if self.args.decoder_freeze:
            decout = [d.detach() for d in decout]

        target = self.patchify(img1)
        recon_img = self.prediction_head(decout[-1])

            
        if self.args.attn_agg:
            output= self.head(decout, img_info, attn_map, intrinsics=intrinsics)

            output['target'] = target
            output['recon_img'] = recon_img
            output['mask1'] = mask1
            

            return output
        

        output = self.head(decout, img_info, attn_map)

        output['target'] = target
        output['recon_img'] = recon_img
        output['mask1'] = mask1
        
        return output