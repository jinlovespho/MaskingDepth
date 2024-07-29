import random
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import datasets

import utils
from einops import rearrange


# seed setting
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed : {seed}")
    

def print_exp_info(train_args):
        print('===================================')
        print('LOAD_CKPT: ', train_args.pretrained_weight_path)
        print('===================================')
        print('DATASET: ', train_args.dataset)
        print('KITTI SPLIT: ', train_args.splits)
        print('-----------------------------------')
        print('BACKBONE_LR: ', train_args.backbone_lr)
        print('ELSE_LR: ', train_args.learning_rate)
        print('BATCH SIZE: ', train_args.batch_size)
        print('MASKING_RATIO: ', train_args.masking_ratio)
        print('-----------------------------------')
        print('EPOCH SAVE FREQ: ', train_args.epoch_save_freq)
        print('LOG TOOL: ', train_args.log_tool)
        print('WANDB EXP NAME: ', train_args.wandb_exp_name)
        print('===================================')

############################################################################## 
########################    model load
############################################################################## 

def model_load(train_args, device):
    model = {}
    params_to_train = []
    

    if train_args.model_info == 'mf_croco_baseline':
        # croco models
        from networks.croco_models.croco_downstream import CroCoDownstreamBinocular, croco_args_from_ckpt
        from networks.croco_models.head_downstream import PixelwiseTaskWithDPT    
        from networks.croco_models.pos_embed import interpolate_pos_embed    
        # monodepth2 pose models
        from networks.monodepth2_models.resnet_encoder import ResnetEncoder
        from networks.monodepth2_models.pose_decoder import PoseDecoder
        
        # load models
        ckpt = torch.load(train_args.pretrained_weight_path, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = (train_args.re_height, train_args.re_width)    # 192 640
        print('CROCO ARGS INFO: '+str(croco_args))
        num_channels = 1
        print(f'Building head PixelwiseTaskWithDPT() with {num_channels} channel(s)')
        head = PixelwiseTaskWithDPT()
        head.num_channels = num_channels
        breakpoint()
        model['depth'] = CroCoDownstreamBinocular(head, **croco_args)
        interpolate_pos_embed(model['depth'], ckpt['model'])
        msg = model['depth'].load_state_dict(ckpt['model'], strict=False)
        # print(msg)
        model["pose_enc"] = ResnetEncoder(18,True,num_input_images=2 )
        model["pose_dec"] = PoseDecoder( model["pose_enc"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        
        # set trainable params
        enc_params, enc_names = [], []
        dec_params, dec_names = [], []
        else_params, else_names = [], []
        for name, param in model['depth'].named_parameters():
            if 'enc_blocks' in name or 'enc_norm' in name:
                enc_params.append(param)
                enc_names.append(name)
            elif 'dec_blocks' in name or 'decoder_embed' in name or 'dec_norm' in name:
                dec_params.append(param)
                dec_names.append(name)
            else:
                else_params.append(param)
                else_names.append(name)
        
        depth_params=list(model['depth'].parameters())
        assert len(depth_params) == len(enc_params) + len(dec_params) + len(else_params), 'CHECK TRAINABLE PARAMS !!'
        
        else_params+=model['pose_enc'].parameters()
        else_params+=model['pose_dec'].parameters()
        
        params_to_train.append( {'params':enc_params, 'lr':train_args.lr*0.1} )
        params_to_train.append( {'params':dec_params, 'lr':train_args.lr*0.1} )
        params_to_train.append( {'params':else_params, 'lr':train_args.lr})
        
    
    elif train_args.model_info == 'mf_croco_try1':
        # croco models
        from networks.croco_models_try1.croco_downstream import CAMapCroCoDownstreamBinocular, croco_args_from_ckpt
        from networks.croco_models_try1.head_downstream import CAMapPixelwiseTaskWithDPT    
        from networks.croco_models_try1.pos_embed import interpolate_pos_embed    
        # monodepth2 pose models
        from networks.monodepth2_models.resnet_encoder import ResnetEncoder
        from networks.monodepth2_models.pose_decoder import PoseDecoder
        
        # load models
        ckpt = torch.load(train_args.pretrained_weight_path, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = (train_args.re_height, train_args.re_width)    # 192 640
        print('CROCO ARGS INFO: '+str(croco_args))
        num_channels = 1
        print(f'Building head PixelwiseTaskWithDPT() with {num_channels} channel(s)')
        head = CAMapPixelwiseTaskWithDPT()
        head.num_channels = num_channels
        breakpoint()
        model['depth'] = CAMapCroCoDownstreamBinocular(head, **croco_args)
        interpolate_pos_embed(model['depth'], ckpt['model'])
        msg = model['depth'].load_state_dict(ckpt['model'], strict=False)
        # print(msg)
        model["pose_enc"] = ResnetEncoder(18,True,num_input_images=2 )
        model["pose_dec"] = PoseDecoder( model["pose_enc"].num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        
        # set trainable params
        enc_params, enc_names = [], []
        dec_params, dec_names = [], []
        else_params, else_names = [], []
        for name, param in model['depth'].named_parameters():
            if 'enc_blocks' in name or 'enc_norm' in name:
                enc_params.append(param)
                enc_names.append(name)
            elif 'dec_blocks' in name or 'decoder_embed' in name or 'dec_norm' in name:
                dec_params.append(param)
                dec_names.append(name)
            else:
                else_params.append(param)
                else_names.append(name)
        
        depth_params=list(model['depth'].parameters())
        assert len(depth_params) == len(enc_params) + len(dec_params) + len(else_params), 'CHECK TRAINABLE PARAMS !!'
        
        else_params+=model['pose_enc'].parameters()
        else_params+=model['pose_dec'].parameters()
        
        params_to_train.append( {'params':enc_params, 'lr':train_args.lr*0.1} )
        params_to_train.append( {'params':dec_params, 'lr':train_args.lr*0.1} )
        params_to_train.append( {'params':else_params, 'lr':train_args.lr})
        
    else:
        print('NO MODEL TO LOAD')
        breakpoint()
    
    # if train_args.load_weight_path is not None:
    #     print('load_weight_path')
    #     model['depth'].load_state_dict(torch.load(train_args.load_weight_path))

    for key, val in model.items():
        model[key] = nn.DataParallel(val)   # set data parallel training
        model[key].to(device)   # put model to cuda
        model[key].train()

    return model, params_to_train


############################################################################## 
########################    data laoder
############################################################################## 

def data_loader(train_args, batch_size, num_workers):  
    # data loader
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                    "kitti_odom": datasets.KITTIOdomDataset,
                    "kitti_depth": datasets.KITTIDepthDataset,
                    "nyu": datasets.NYUDataset,
                    "virtual_kitti": datasets.Virtual_Kitti,
                    "kitti_depth_multiframe":datasets.KITTIDepthMultiFrameDataset }

    dataset = datasets_dict[train_args.dataset]
    fpath = os.path.join(os.path.dirname(__file__), "splits", train_args.splits, "{}_files.txt")
    
    train_filenames = utils.readlines(fpath.format("train"))
    val_filenames   = utils.readlines(fpath.format("val"))
    
    print('DATASET: ', dataset)
    # breakpoint()
    
    train_ds = dataset(train_args.data_path, train_filenames, train_args.re_height, train_args.re_width, 
                       train_args.frame_ids, 4, is_train=True, img_ext=train_args.img_ext)
    
    val_ds =   dataset(train_args.data_path, val_filenames, train_args.re_height, train_args.re_width, 
                       train_args.frame_ids, 4, is_train=True, img_ext=train_args.img_ext)
    
    
    # if train_args.dataset == 'kitti_depth_multiframe':
    #     train_dataset = dataset(train_args.data_path, train_filenames, train_args.re_height, train_args.re_width, use_box = True, 
    #                              gt_num = -1, is_train=True, img_ext=train_args.img_ext, num_prev_frame=train_args.num_prev_frame)
        
    #     val_dataset = dataset(train_args.data_path, val_filenames, train_args.re_height, train_args.re_width, use_box = True, 
    #                            gt_num = -1, is_train=False, img_ext=train_args.img_ext, num_prev_frame=train_args.num_prev_frame)
    
    # else:
    #     train_dataset = dataset(train_args.data_path, train_filenames, train_args.re_height, train_args.re_width, use_box = True, 
    #                             gt_num = -1, is_train=True, img_ext=train_args.img_ext)
    #     val_dataset = dataset(train_args.data_path, val_filenames, train_args.re_height, train_args.re_width, use_box = True, 
    #                             gt_num = -1, is_train=False, img_ext=train_args.img_ext)
    
    train_loader = DataLoader(train_ds, batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size, False, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_ds, val_ds, train_loader, val_loader