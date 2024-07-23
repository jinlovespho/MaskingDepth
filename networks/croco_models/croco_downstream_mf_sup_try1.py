# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# CroCo model for downstream tasks
# --------------------------------------------------------

import torch

from .croco import CroCoNet

class CroCoDownstreamBinocular_MF_Sup_Try1(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features 
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super().__init__(**kwargs)
        head.setup(self)
        self.head = head

    def _set_mask_generator(self, *args, **kwargs):
        """ No mask generator """
        return

    def _set_mask_token(self, *args, **kwargs):
        """ No mask token """
        self.mask_token = None
        return

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head for downstream tasks, define your own head """
        return
        
    def encode_image_pairs(self, img1, img2, return_all_blocks=False):
        """ run encoder for a pair of images
            it is actually ~5% faster to concatenate the images along the batch dimension 
             than to encode them separately
        """
        ## the two commented lines below is the naive version with separate encoding
        #out, pos, _ = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
        #out2, pos2, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False)
        ## and now the faster version
        
        '''
            img1, img2: b,3,192,640
        '''
        
        # breakpoint()
        out, pos, _ = self._encode_image( torch.cat( (img1,img2), dim=0), do_mask=False, return_all_blocks=return_all_blocks )
        
        if return_all_blocks:
            out1, out2 = list(map(list, zip(*[o.chunk(2, dim=0) for o in out])))  

        pos1,pos2 = pos.chunk(2, dim=0)           
        return out1, out2, pos1, pos2

    def forward(self, img1, img2):
        # breakpoint()
        B, C, H, W = img1.size()    # b 3 192 640
        img_info = {'height': H, 'width': W}    # 192 640
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks     # True
        
        # encode img1, img2 
        out1, out2, pos1, pos2 = self.encode_image_pairs(img1, img2, return_all_blocks=return_all_blocks)

        # decode img1, img2
        if return_all_blocks:
            dec1_out, dec1_samap, dec1_camap = self._decoder(out1[-1], pos1, None, out2[-1], pos2, return_all_blocks=return_all_blocks)
            # dec2_out, dec2_samap, dec2_camap = self._decoder(out2[-1], pos2, None, out1[-1], pos1, return_all_blocks=return_all_blocks)
            
            dec_out1 = out1+dec1_out    # 리스트끼리 덧셈. 그냥 append하는 것, 앞쪽에 out 즉 encoder output이 앞쪽으로 오도록
            # dec_out2 = out2+dec2_out 
            
        # pass to head
        return self.head(dec_out1, img_info)
    
    
