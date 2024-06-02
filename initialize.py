import random
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import datasets
import networks
from networks.my_models import *
import utils
from einops import rearrange

import timm

FULL  = 0
FRONT = 1
BACK  = 2

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

############################################################################## 
########################    model load
############################################################################## 

def baseline_model_load(train_cfg, device):
    model = {}
    parameters_to_train = []
    
    # try1 - unet w/o skip
    if train_cfg.model.baseline == 'my_try1': 
        print(train_cfg.model.enc_name)
        breakpoint()
        from networks.my_models.my_try1_unet import My_Unet
        model['depth'] = My_Unet(train_cfg)
        
    # try2 - unet w/ skip
    elif train_cfg.model.baseline == 'my_try2': 
        print(train_cfg.model.enc_name)
        breakpoint()
        from networks.my_models.my_try2_unet_skip import My_Unet_Skip
        model['depth'] = My_Unet_Skip(train_cfg)
        
    # try3 - vitB_16 encoder
    elif train_cfg.model.baseline == 'my_try3': 
        print(train_cfg.model.enc_name)
        breakpoint()
        from networks.my_models.my_try3_unet_vitb import My_Unet_Enc_VitB
        model['depth'] = My_Unet_Enc_VitB(train_cfg)
        
    # try3a - vitB_14_dinov2 encoder
    elif train_cfg.model.baseline == 'my_try3a': 
        print(train_cfg.model.enc_name)
        breakpoint()
        from networks.my_models.my_try3a_unet_vitb_14_dino import My_Unet_Enc_VitB_Dinov2
        model['depth'] = My_Unet_Enc_VitB_Dinov2(train_cfg)
        
    # try3b - vitB_8 encoder
    elif train_cfg.model.baseline == 'my_try3b': 
        print(train_cfg.model.enc_name)
        breakpoint()
        from networks.my_models.my_try3b_unet_vitb_8 import My_Unet_Enc_VitB_8
        model['depth'] = My_Unet_Enc_VitB_8(train_cfg)


    # JINLOVESPHO costvolume_try1
    elif train_cfg.model.baseline == 'costvolume_try1':
        
        if train_cfg.model.vit_type == 'vit_base':
            print('ENCODER: vit_base')
            enc_layers=12
            enc_hidden_dim=768
            enc_mlp_dim=3072
            enc_heads=12
        
        elif train_cfg.model.vit_type == 'vit_large':
            print('ENCODER: vit_large')
            enc_layers=24
            enc_hidden_dim=1024
            enc_mlp_dim=4096
            enc_heads=16
        
        else:
            print('vit type not valid')

        v = networks.ViT_Multiframe(    image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = enc_hidden_dim,
                                        depth = enc_layers,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = enc_heads,
                                        mlp_dim = enc_mlp_dim,
                                        num_prev_frame=train_cfg.model.num_prev_frame,
                                        croco = (train_cfg.model.pretrained_weight == 'croco'))
        
        if train_cfg.model.pretrained_weight == 'croco':
            
            if train_cfg.model.vit_type == 'vit_base':
                croco_weight = torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth', map_location=device)
            elif train_cfg.model.vit_type == 'vit_large':
                croco_weight = torch.load('../pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth', map_location=device)

            loaded_weight = {}
            
            for key, value in v.state_dict().items():
                if 'transformer' in key:
                    if '0.norm' in key:
                        # breakpoint()
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.norm1.{key.split(".")[-1]}']
                    elif 'qkv' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.attn.qkv.{key.split(".")[-1]}']
                    elif 'to_out' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.attn.proj.{key.split(".")[-1]}']
                    elif '1.norm' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.norm2.{key.split(".")[-1]}']
                    elif 'fn.net.0' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.mlp.fc1.{key.split(".")[-1]}']
                    elif 'fn.net.3' in key:
                        loaded_weight[key] = croco_weight['model'][f'enc_blocks.{key.split(".")[2]}.mlp.fc2.{key.split(".")[-1]}']
                    
                elif 'to_patch_embedding' in key:
                    loaded_weight[key] = croco_weight['model'][f'patch_embed.proj.{key.split(".")[-1]}']

                else:
                    print(key)
                    loaded_weight[key] = v.state_dict()[key]
        
        else:
            loaded_weight = torch.load("../../MaskingDepth/vit_base_384.pth", map_location=device)
        
            for key, value in v.state_dict().items():
                if key not in loaded_weight.keys():
                    loaded_weight[key] = loaded_weight['pos_embedding']
        
        is_load_complete = v.load_state_dict(loaded_weight)
        print(is_load_complete)
        v.resize_pos_embed(192,640,device)

        breakpoint()
        model['depth'] = networks.mask_dpt_multiframe_croco_costvolume_try1.Masked_DPT_Multiframe_Croco_Costvolume_Try1(encoder=v,
                        max_depth = train_cfg.model.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=enc_hidden_dim,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=train_cfg.model.num_prev_frame,
                        masking_ratio=train_cfg.model.masking_ratio,
                        num_frame_to_mask=train_cfg.model.num_frame_to_mask,
                        cross_attn_depth = train_cfg.model.cross_attn_depth,
                        croco = (train_cfg.model.pretrained_weight == 'croco'),
                        )
    
    else:
        pass
    
    if train_cfg.model.load_weight:
        print("Loading Network weights")
        depth_file = os.path.join(train_cfg.model.weight_path, 'depth.pth')
        if os.path.isfile(depth_file):
            print("Success load depth weight")
            model_load_dict = torch.load(depth_file, map_location=device)
            model['depth'].load_state_dict(model_load_dict)
        else:
            print(f"Dose not exist {depth_file}")

    for key, val in model.items():
        model[key] = nn.DataParallel(val)
        model[key].to(device)
        model[key].train()
        parameters_to_train += list(val.parameters())

    return model, parameters_to_train


############################################################################## 
########################    data laoder
############################################################################## 

def data_loader(data_cfg, batch_size, num_workers):  
    # data loader
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                    "kitti_odom": datasets.KITTIOdomDataset,
                    "kitti_depth": datasets.KITTIDepthDataset,
                    "nyu": datasets.NYUDataset,
                    "virtual_kitti": datasets.Virtual_Kitti,
                    "kitti_depth_multiframe":datasets.KITTIDepthMultiFrameDataset }
    # breakpoint()
    dataset = datasets_dict[data_cfg.dataset]
    fpath = os.path.join(os.path.dirname(__file__), "splits", data_cfg.splits, "{}_files.txt")

    train_filenames = utils.readlines(fpath.format("train"))
    val_filenames   = utils.readlines(fpath.format("val"))
    
    print('DATASET: ', dataset)
    # breakpoint()
    if data_cfg.dataset == 'kitti_depth_multiframe':
        train_dataset = dataset(data_cfg.data_path, train_filenames, data_cfg.height, data_cfg.width, use_box = data_cfg.use_box, 
                                 gt_num = -1, is_train=True, img_ext=data_cfg.img_ext, num_prev_frame=data_cfg.num_prev_frame)
        
        val_dataset = dataset(data_cfg.data_path, val_filenames, data_cfg.height, data_cfg.width, use_box = data_cfg.use_box, 
                               gt_num = -1, is_train=False, img_ext=data_cfg.img_ext, num_prev_frame=data_cfg.num_prev_frame)
    
    else:
        train_dataset = dataset(data_cfg.data_path, train_filenames, data_cfg.height, data_cfg.width, use_box = data_cfg.use_box, 
                                gt_num = -1, is_train=True, img_ext=data_cfg.img_ext)
        val_dataset = dataset(data_cfg.data_path, val_filenames, data_cfg.height, data_cfg.width, use_box = data_cfg.use_box, 
                                gt_num = -1, is_train=False, img_ext=data_cfg.img_ext)
    
    train_loader = DataLoader(train_dataset, batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size, False, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_dataset, val_dataset, train_loader, val_loader