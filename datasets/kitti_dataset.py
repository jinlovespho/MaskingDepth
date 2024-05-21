# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import skimage.transform
import numpy as np
import PIL.Image as pil
import torch

from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset

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



class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip=False):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    # BJB EDIT
    def get_Bbox(self, folder, frame_index, side, do_flip = False):

        #file path
        f_str = "{:010d}_box.txt".format(frame_index)
        
        box_path = os.path.join(self.data_path, folder,
            "box_0{}".format(self.side_map[side]), f_str)
        
        # ForkedPdb().set_trace()

        #read file
        with open(box_path, 'r') as f:
            boxes = []
            
            for line in f.readlines():
                param = line.split(' ')
                param[0] = int(param[0])        # class
                param[1] = float(param[1])      # x1 left
                param[2] = float(param[2])      # y1 top
                param[3] = float(param[3])      # x2 right
                param[4] = float(param[4])      # y2 bottom
                param[5] = float(param[5])      # confidence score
                
                if do_flip:#box flip
                    max_x = round(1 - param[1],4)
                    min_x = round(1 - param[3],4)
                    param[1] = min_x
                    param[3] = max_x
                
                # width,height > 0.2 인 어느 정도 크기가 있는 bbox만 가져오고 싶은 것
                # if param[3]-param[1] > 0.2 or param[4]-param[2] > 0.2:
                #     boxes.append(param[:-1])   
                boxes.append(param)
             
        # ForkedPdb().set_trace()       
        # fix number of bboxes
        self.MAX_BOX_NUM=50
        if len(boxes) >= self.MAX_BOX_NUM:
            return torch.tensor(boxes[:self.MAX_BOX_NUM])
        else:        
            return torch.cat((torch.tensor(boxes), - torch.ones((self.MAX_BOX_NUM - len(boxes)), 6)), dim=0)


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip = False):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip = False):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIDepthMultiFrameDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthMultiFrameDataset, self).__init__(*args, **kwargs)
        self.num_prev_frame = kwargs['num_prev_frame']

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip = False):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)
        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    
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
     
        
        # inputs = {}
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
            frame_index = int(line[1])
            side = line[2]          
            
            start_idx = frame_index 
            num_frames = self.num_prev_frame+1
            inputs_lst = []
            
            if start_idx < 5:
                start_idx = 5 
                                 
            for _ in range(num_frames):             
                inputs={}
                inputs['curr_folder']=folder
                inputs['curr_frame']=start_idx       
                inputs["color"] = self.get_color(folder, start_idx, side, do_flip)
                inputs["box"] = self.get_Bbox(folder, start_idx, side, do_flip)
                
                # MAGNET
                # preparation for extrinsic matrix
                dataset_path = self.data_path
                date = line[0].split('/')[0]
                drive = line[0].split('/')[1].split('_')[-2] 
                p_data = pykitti.raw(dataset_path, date, drive, frames=[start_idx], imtype='jpg')
        
                # ForkedPdb().set_trace()
                # cam intrinsics
                cam_intrins = self.get_cam_intrinsics(p_data)
                inputs['ray'] = cam_intrins['unit_ray_array_2D']
                inputs['intM'] = cam_intrins['intM']
                # ForkedPdb().set_trace()
                
                # cam extrinsic (pose)
                pose = p_data.oxts[0].T_w_imu
                M_imu2cam = p_data.calib.T_cam2_imu
                extM = np.matmul(M_imu2cam, np.linalg.inv(pose))
                inputs['extM'] = extM
                
                # ForkedPdb().set_trace()  
                
                K = self.K.copy()
                K[0, :] *= self.width 
                K[1, :] *= self.height

                inv_K = np.linalg.pinv(K)

                inputs["K"] = torch.from_numpy(K)
                inputs["inv_K"] = torch.from_numpy(inv_K)

                stereo_T = np.eye(4, dtype=np.float32)
                baseline_sign = -1 if do_flip else 1
                side_sign = -1 if side == "l" else 1
                stereo_T[0, 3] = side_sign * baseline_sign * 0.1

                inputs["stereo_T"] = torch.from_numpy(stereo_T)
                
                self.preprocess(inputs)
            
                if self.load_depth:
                    depth_gt = self.get_depth(folder, start_idx, side, do_flip)
                    inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
                    inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

                inputs_lst.append(inputs)
                
                # for loop 안에서 start_idx 를 줄여가서 이전 frame 들을 가져오는 것
                start_idx -= 1
                
                if start_idx < 5:
                    start_idx = 5  
            
            # ForkedPdb().set_trace()
            return inputs_lst           # For multiframe kitti dataset 
          
        return inputs
            