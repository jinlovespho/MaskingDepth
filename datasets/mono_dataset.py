# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
from math import degrees

import os
import random
from tabnanny import check
# from BLP_depth.loss import train_mode
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.nn.functional as F

import torch.utils.data as data
from torchvision import transforms
import torchvision.utils
from  .autoaugment import rand_augment_transform, Cutout, _rotate_level_to_arg
import pykitti

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

ROLL    = 0
ROTATE  = 1
CUTOUT  = 2

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()
        
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        
        self.frame_idxs = frame_idxs
        
        self.is_train = is_train
        self.img_ext = img_ext
        
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        
        self.MAX_BOX_NUM = 8
        
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
            
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        # self.resize = transforms.Resize((self.height, self.width), interpolation=self.interp)
        self.load_depth = self.check_depth()
        
        self.rand_aug = rand_augment_transform(config_str = 'rand-n{}-m{}-mstd0.5'.format(3, 5), hparams={})
        
        # Magnet 에서 가져와 O
        # image resolution
        # 여기를 제대로 한지 모르겠네.. 나중에 다시 봐야 O
        self.img_H = 375
        self.img_W = 1242
        self.dpv_H = self.height
        self.dpv_W = self.width
        
        # ray array
        self.ray_array = self.get_ray_array()
                
                
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    # def preprocess(self, inputs):
    #     """Resize colour images to the required scales and augment if required

    #     We create the color_aug object in advance and apply the same augmentation to all
    #     images in this item. This ensures that all images input to the pose network receive the
    #     same augmentation.
    #     """
            
    #     origin = inputs["color"]
    #     do_weak = self.is_train and random.random() > 0.5
    #     Geo_aug = random.randint(0,3)
        
    #     #weak aug
    #     if do_weak:
    #         weak_aug = transforms.ColorJitter(
    #                         self.brightness, self.contrast, self.saturation, self.hue)
    #         inputs["color"] = self.to_tensor(self.resize(weak_aug(inputs["color"])))
    #         torchvision.utils.save_image(inputs["color"], "./aaa.png")
    #     else:
    #         inputs["color"] = self.to_tensor(self.resize(inputs["color"]))
                        
    #     #strong aug
    #     inputs["color_aug"] = self.to_tensor(self.resize(self.rand_aug(origin)))        
    
    def __len__(self):
        return len(self.filenames)

    # ray array used to back-project depth-map into camera-centered coordinates
    def get_ray_array(self):
        ray_array = np.ones((self.dpv_H, self.dpv_W, 3))
        x_range = np.arange(self.dpv_W)
        y_range = np.arange(self.dpv_H)
        x_range = np.concatenate([x_range.reshape(1, self.dpv_W)] * self.dpv_H, axis=0)
        y_range = np.concatenate([y_range.reshape(self.dpv_H, 1)] * self.dpv_W, axis=1)
        ray_array[:, :, 0] = x_range + 0.5
        ray_array[:, :, 1] = y_range + 0.5
        return ray_array
    
    # get camera intrinscs - jinlovespho - added from magnet
    def get_cam_intrinsics(self, p_data):

        raw_img_size = p_data.get_cam2(0).size
        raw_W = int(raw_img_size[0])
        raw_H = int(raw_img_size[1])

        top_margin = int(raw_H - 352)
        left_margin = int((raw_W - 1216) / 2)

        # original intrinsic matrix (4X4)
        IntM_ = p_data.calib.K_cam2
        
        # ForkedPdb().set_trace()

        # updated intrinsic matrix
        IntM = np.zeros((3, 3))
        IntM[2, 2] = 1.
        IntM[0, 0] = IntM_[0, 0] * (self.dpv_W / float(self.img_W))
        IntM[1, 1] = IntM_[1, 1] * (self.dpv_H / float(self.img_H))
        IntM[0, 2] = (IntM_[0, 2] - left_margin) * (self.dpv_W / float(self.img_W))
        IntM[1, 2] = (IntM_[1, 2] - top_margin) * (self.dpv_H / float(self.img_H))

        # pixel to ray array
        pixel_to_ray_array = np.copy(self.ray_array)
        pixel_to_ray_array[:, :, 0] = ((pixel_to_ray_array[:, :, 0] * (self.img_W / float(self.dpv_W)))
                                       - IntM_[0, 2] + left_margin) / IntM_[0, 0]
        pixel_to_ray_array[:, :, 1] = ((pixel_to_ray_array[:, :, 1] * (self.img_H / float(self.dpv_H)))
                                       - IntM_[1, 2] + top_margin) / IntM_[1, 1]

        pixel_to_ray_array_2D = np.reshape(np.transpose(pixel_to_ray_array, axes=[2, 0, 1]), [3, -1])
        pixel_to_ray_array_2D = torch.from_numpy(pixel_to_ray_array_2D.astype(np.float32))

        cam_intrinsics = {
            'unit_ray_array_2D': pixel_to_ray_array_2D,
            'intM': torch.from_numpy(IntM.astype(np.float32))
        }
        return cam_intrinsics
    
    
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.
 
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:
            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,

            ("color")                               for raw colour images,
            ("color_aug")                           for augmented colour images,
            ("K") or ("inv_K")                      for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.
            "box"                                   for using bounding box
        """
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        if type(self).__name__ in "CityscapeDataset":
            folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
            inputs.update(self.get_color(folder, frame_index, side, do_flip))

        elif type(self).__name__ in "NYUDataset" or type(self).__name__ in "Virtual_Kitti":
            if type(self).__name__ in "NYUDataset":
                split_com = '/'
            else:
                split_com = ' '
           
            folder = os.path.join(*self.filenames[index].split(split_com)[:-1])
            
            if self.is_train:
                frame_index = int(self.filenames[index].split(split_com)[-1])
                side = None
                inputs["color"] = self.get_color(folder, frame_index, side, do_flip)
            else:
                frame_index = self.filenames[index].split(split_com)[-1]
                side= None
                inputs["color"] = self.get_color(folder, frame_index, side, do_flip)

        else:
            line = self.filenames[index].split()
            folder = line[0]
            
            if len(line) == 3:
                frame_index = int(line[1])
            else:
                frame_index = 0

            if len(line) == 3:
                side = line[2]
            else:
                side = None

            for i in self.frame_idxs:
                if i == "s":
                    other_side = {"r": "l", "l": "r"}[side]
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
                else:
                    inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)


            # adjusting intrinsics to match each scale in the pyramid
            for scale in range(self.num_scales):
                K = self.K.copy()

                K[0, :] *= self.width // (2 ** scale)
                K[1, :] *= self.height // (2 ** scale)

                inv_K = np.linalg.pinv(K)

                inputs[("K", scale)] = torch.from_numpy(K)
                inputs[("inv_K", scale)] = torch.from_numpy(inv_K)
                        
            if do_color_aug:
                color_aug = transforms.ColorJitter( self.brightness, self.contrast, self.saturation, self.hue)
            else:
                color_aug = (lambda x: x)

            # ForkedPdb().set_trace() 
            self.preprocess(inputs, color_aug)  
            
            for i in self.frame_idxs:
                del inputs[("color", i, -1)]
                del inputs[("color_aug", i, -1)]

            if self.load_depth:
                depth_gt = self.get_depth(folder, frame_index, side, do_flip)
                inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
        
            # from torchvision.utils import save_image 
            # save_image(inputs['depth_gt'], '../d1.jpg')
            # save_image(inputs['depth_gt'], '../d1_n.jpg', normalize=True)
            # ForkedPdb().set_trace()
            
            if "s" in self.frame_idxs:
                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side == "l" else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1

                inputs["stereo_T"] = torch.from_numpy(stereo_T)
            
            
            # JINLOVESPHO 여기서부터 수정 위에까지는 monodepth2
            frame_index = int(line[1])
            side = line[2]
            # inputs["color"] = self.get_color(folder, frame_index, side, do_flip)
            inputs["box"] = self.get_Bbox(folder, frame_index, side, do_flip)
            inputs['curr_folder']=folder
            inputs['curr_frame']=frame_index
            
            # MAGNET
            # preparation for extrinsic matrix
            dataset_path = self.data_path
            date = line[0].split('/')[0]
            drive = line[0].split('/')[1].split('_')[-2] 
            p_data = pykitti.raw(dataset_path, date, drive, frames=[frame_index], imtype='jpg')
     
            # cam intrinsics
            cam_intrins = self.get_cam_intrinsics(p_data)
            inputs['ray'] = cam_intrins['unit_ray_array_2D']
            inputs['intM'] = cam_intrins['intM']

            # cam extrinsic (pose)
            pose = p_data.oxts[0].T_w_imu
            M_imu2cam = p_data.calib.T_cam2_imu
            extM = np.matmul(M_imu2cam, np.linalg.inv(pose))
            inputs['extM'] = extM
        
          
        return inputs
    

    # def __getitem__(self, index):
    #     """Returns a single training item from the dataset as a dictionary.
 
    #     Values correspond to torch tensors.
    #     Keys in the dictionary are either strings or tuples:
    #         ("color", <frame_id>, <scale>)          for raw colour images,
    #         ("color_aug", <frame_id>, <scale>)      for augmented colour images,

    #         ("color")                               for raw colour images,
    #         ("color_aug")                           for augmented colour images,
    #         ("K") or ("inv_K")                      for camera intrinsics,
    #         "stereo_T"                              for camera extrinsics, and
    #         "depth_gt"                              for ground truth depth maps.
    #         "box"                                   for using bounding box
    #     """
    #     inputs = {}
    #     do_flip = self.is_train and random.random() > 0.5

    #     if type(self).__name__ in "CityscapeDataset":
    #         folder, frame_index, side = self.index_to_folder_and_frame_idx(index)
    #         inputs.update(self.get_color(folder, frame_index, side, do_flip))

    #     elif type(self).__name__ in "NYUDataset" or type(self).__name__ in "Virtual_Kitti":
    #         if type(self).__name__ in "NYUDataset":
    #             split_com = '/'
    #         else:
    #             split_com = ' '
           
    #         folder = os.path.join(*self.filenames[index].split(split_com)[:-1])
            
    #         if self.is_train:
    #             frame_index = int(self.filenames[index].split(split_com)[-1])
    #             side = None
    #             inputs["color"] = self.get_color(folder, frame_index, side, do_flip)
    #         else:
    #             frame_index = self.filenames[index].split(split_com)[-1]
    #             side= None
    #             inputs["color"] = self.get_color(folder, frame_index, side, do_flip)

    #     else:
    #         line = self.filenames[index].split()
    #         folder = line[0]
    #         frame_index = int(line[1])
    #         side = line[2]
    #         inputs["color"] = self.get_color(folder, frame_index, side, do_flip)
    #         inputs["box"] = self.get_Bbox(folder, frame_index, side, do_flip)
    #         inputs['curr_folder']=folder
    #         inputs['curr_frame']=frame_index
            
    #         # MAGNET
    #         # preparation for extrinsic matrix
    #         dataset_path = self.data_path
    #         date = line[0].split('/')[0]
    #         drive = line[0].split('/')[1].split('_')[-2] 
    #         p_data = pykitti.raw(dataset_path, date, drive, frames=[frame_index], imtype='jpg')
     
    #         # ForkedPdb().set_trace()
    #         # cam intrinsics
    #         cam_intrins = self.get_cam_intrinsics(p_data)
    #         inputs['ray'] = cam_intrins['unit_ray_array_2D']
    #         inputs['intM'] = cam_intrins['intM']
    #         # ForkedPdb().set_trace()
            
    #         # cam extrinsic (pose)
    #         pose = p_data.oxts[0].T_w_imu
    #         M_imu2cam = p_data.calib.T_cam2_imu
    #         extM = np.matmul(M_imu2cam, np.linalg.inv(pose))
    #         inputs['extM'] = extM
        
    #         # ForkedPdb().set_trace()
            
    #         K = self.K.copy()
    #         K[0, :] *= self.width 
    #         K[1, :] *= self.height 

    #         inv_K = np.linalg.pinv(K)

    #         inputs["K"] = torch.from_numpy(K)
    #         inputs["inv_K"] = torch.from_numpy(inv_K)

    #         stereo_T = np.eye(4, dtype=np.float32)
    #         baseline_sign = -1 if do_flip else 1
    #         side_sign = -1 if side == "l" else 1
    #         stereo_T[0, 3] = side_sign * baseline_sign * 0.1

    #         inputs["stereo_T"] = torch.from_numpy(stereo_T)
            
            
    #     self.preprocess(inputs)
        
    #     if self.load_depth:
    #         depth_gt = self.get_depth(folder, frame_index, side, do_flip)
    #         inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
    #         inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

    #     return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    
