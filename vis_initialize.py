import random
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import datasets

import networks.monodepth2_networks

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
        print('LOAD_CKPT: ', train_args.load_weight_path)
        print('===================================')
        print('DATASET: ', train_args.dataset)
        print('KITTI SPLIT: ', train_args.splits)
        print('-----------------------------------')
        print('BATCH SIZE: ', train_args.batch_size)
        print('-----------------------------------')
        print('LOG TOOL: ', train_args.log_tool)
        print('WANDB EXP NAME: ', train_args.wandb_exp_name)
        print('===================================')

############################################################################## 
########################    model load
############################################################################## 

def model_load(train_args, device):
    model = {}
    params_to_train = []
    
    if train_args.model_info == 'DPT':
        v = networks.vit.ViT( image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                              patch_size = 16,
                              num_classes = 1000,
                              dim = 768,
                              depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                              heads = 12,
                              mlp_dim = 3072)
        is_well_loaded=v.load_state_dict(torch.load("../pretrained_weights/vit_base_384.pth"))
        print(is_well_loaded)
        v.resize_pos_embed(192,640)

        breakpoint()
        model['depth'] = networks.Masked_DPT(encoder=v,
                        max_depth = train_args.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project')      # DPT 에서는 cls token = readout token 이라고 부르고 projection으로 cls token 처리 
        
    elif train_args.model_info == 'DPT_H':
        v = networks.ViT(image_size = (384,384),
                        patch_size = 16,
                        num_classes = 1000,
                        dim = 768,
                        depth = 12,
                        heads = 12,
                        mlp_dim = 3072,
                        hybrid = True)
            
        model['depth'] = networks.Masked_DPT_hybrid(encoder=v,
                        features=[256, 512, 768, 768],
                        hooks=[0, 1, 8, 11] ,
                        max_depth = train_args.max_depth,
                        use_readout='project')
                                
    elif train_args.model_info == 'monodepth2':
        resnet_encoder = networks.ResnetEncoder(50, True, mask_layer=3)
        depth_decoder = networks.DepthDecoder(num_ch_enc=resnet_encoder.num_ch_enc, scales=range(4))
        model['depth'] = networks.Monodepth(resnet_encoder, depth_decoder, max_depth = train_args.max_depth)
        
    
    # crocov2_stereo_flow
    elif train_args.model_info == 'vis_mf_sup_crocov2_baseline':
        from networks.croco_models.croco_downstream import CroCoDownstreamBinocular
        from networks.croco_models.head_downstream import PixelwiseTaskWithDPT
        from networks.croco_models.pos_embed import interpolate_pos_embed
        from networks.mf_sup_crocov2_baseline import MF_Sup_CrocoV2_Baseline
        
        # debug purpose - check pretrained weights
        pw1= torch.load('../pretrained_weights/crocostereo.pth')
        pw2= torch.load('../pretrained_weights/CroCo_V2_ViTBase_SmallDecoder.pth')
        pw3= torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth')
        pw4= torch.load('../pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth')    
        
        # load pretrained weight
        ckpt = torch.load(train_args.pretrained_weight_path)
        if train_args.pretrained_weight_path == '../pretrained_weights/crocostereo.pth':     
            croco_args = ckpt['args'].croco_args
            head_channel=2
            is_strict=True
        else:
            croco_args = ckpt['croco_kwargs']
            head_channel=1
            is_strict=False
            
        croco_args['img_size'] = (train_args.re_height, train_args.re_width)
        croco_head = PixelwiseTaskWithDPT()
        croco_head.num_channels = head_channel     # 1 for only prediction, 2 for pred+confidence
        croco_model = CroCoDownstreamBinocular(head=croco_head, **croco_args)
        interpolate_pos_embed(croco_model, ckpt['model'])   
        msg = croco_model.load_state_dict(ckpt['model'], strict=is_strict)
        print('CROCO_WEIGHT_WELL_LOADED: ', msg)
        print(croco_args)
        
        # show experiment info in terminal
        train_args=vars(train_args)     # vars ! change ~ to dict type
        train_args.update(croco_args)
        train_args = argparse.Namespace(**train_args)   # dict back to namespace
        print_exp_info(train_args)
        
        n_param = sum(i.numel() for i in croco_model.parameters() ) / 1e6
        print(f'NUM_PARAM: {n_param}M ')
        train_args.n_param=n_param
        
        model['depth'] = MF_Sup_CrocoV2_Baseline( model=croco_model )   
        load_weight = torch.load(train_args.load_weight_path)
        is_well_loaded2 = model['depth'].load_state_dict(load_weight, strict=False)
        print('TRAINED WEIGHTS WELL LOADED: ', is_well_loaded2)
        # breakpoint()
        
    
    # crocov2_try1
    elif train_args.model_info == 'mf_sup_crocov2_try1':
        from networks.croco_models.croco_downstream_mf_sup_try1 import CroCoDownstreamBinocular_MF_Sup_Try1
        from networks.croco_models.head_downstream_mf_sup_try1 import PixelwiseTaskWithDPT_MF_Sup_Try1
        from networks.croco_models.pos_embed import interpolate_pos_embed
        from networks.mf_sup_crocov2_try1 import MF_Sup_CrocoV2_Try1

        # load pretrained weight
        ckpt = torch.load(train_args.pretrained_weight_path)
        if train_args.pretrained_weight_path == '../pretrained_weights/crocostereo.pth':     
            croco_args = ckpt['args'].croco_args
            head_channel=2
            is_strict=True
        else:
            croco_args = ckpt['croco_kwargs']
            head_channel=1
            is_strict=False

        croco_args['img_size'] = (train_args.re_height, train_args.re_width)
        croco_head = PixelwiseTaskWithDPT_MF_Sup_Try1()
        croco_head.num_channels = head_channel     # 1 for only prediction, 2 for pred+confidence
        croco_model = CroCoDownstreamBinocular_MF_Sup_Try1(head=croco_head, **croco_args)
        interpolate_pos_embed(croco_model, ckpt['model'])   
        msg = croco_model.load_state_dict(ckpt['model'], strict=is_strict)
        print('CROCO_WEIGHT_WELL_LOADED: ', msg)
        print(croco_args)
        
        # show experiment info in terminal
        train_args=vars(train_args)     # vars ! change ~ to dict type
        train_args.update(croco_args)
        train_args = argparse.Namespace(**train_args)   # dict back to namespace
        print_exp_info(train_args)
        
        n_param = sum(i.numel() for i in croco_model.parameters() ) / 1e6
        print(f'NUM_PARAM: {n_param}M ')
        train_args.n_param=n_param
        
        # breakpoint()
        model['depth'] = MF_Sup_CrocoV2_Try1( model=croco_model )      
        
        model_params = [param for name, param in model['depth'].model.named_parameters()]
        backbone_params = [param for name, param in model['depth'].model.named_parameters() if 'enc_block' in name]
        else_params = [param for name, param in model['depth'].model.named_parameters() if 'enc_block' not in name]
        
        params_to_train.append( {'params':backbone_params, 'lr':train_args.backbone_lr} )
        params_to_train.append( {'params':else_params, 'lr':train_args.lr})

        # tmp1 = [n for n,p in model['depth'].model.named_parameters()]
        # tmp2 = [n for n,p in model['depth'].model.named_modules()]
        t1 = len(model_params)
        t2 = len(backbone_params)
        t3 = len(else_params)
        assert t1 == t2+t3, 'check params_to_train'
        
    else:
        pass
    
    for key, val in model.items():
        model[key].to(device)
        model[key].eval()

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