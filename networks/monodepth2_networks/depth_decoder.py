# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.num_ch_enc = num_ch_enc

        # decoder
        convs = []
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            convs.append(ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            convs.append(ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            convs.append(Conv3x3(self.num_ch_dec[s], num_output_channels))

        self.decoder = nn.ModuleList(convs)
        self.sigmoid = nn.Sigmoid()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input_features):
        
        breakpoint()
        
        outputs = {}
        
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.decoder[-2 * i + 8](x)

            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            x = torch.cat(x, 1)
            x = self.decoder[-2 * i + 9](x)

            outputs[('d_feature', i)] = x

            if i in self.scales:
                outs = self.decoder[10 + i](x)
                outputs[("disp", i)] = F.sigmoid(outs)
                
        '''
        (10): (conv): Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1))
        (11): (conv): Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1))
        (12): (conv): Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1))
        (13): (conv): Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1))

        outputs['disp',0] torch.Size([1, 1, 192, 640])
        outputs['disp',1] torch.Size([1, 1, 96, 320])
        outputs['disp',2] torch.Size([1, 1, 48, 160])
        outputs['disp',3] torch.Size([1, 1, 24, 80])
        '''

        return outputs