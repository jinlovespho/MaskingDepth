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
        print('DATASET: ', train_args.dataset)
        print('KITTI SPLIT: ', train_args.splits)
        print('-----------------------------------')
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
    parameters_to_train = []

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
        
    
    # JINLOVESPHO sf_baseline
    elif train_args.model_info == 'sf_baseline':
        
        v = networks.vit.ViT( image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                              patch_size = 16,
                              num_classes = 1000,
                              dim = 768,
                              depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                              heads = 12,
                              mlp_dim = 3072)
        
        if train_args.pretrained_weight == 'vit_base_384':
            is_well_loaded=v.load_state_dict(torch.load("../pretrained_weights/vit_base_384.pth"))
            print(is_well_loaded)
            
        v.resize_pos_embed(192,640)
        
        # show experiment info in terminal
        print_exp_info(train_args)

        breakpoint()
        model['depth'] = networks.SF_Depth_Baseline(   encoder=v,
                                                max_depth = train_args.max_depth,
                                                features=[96, 192, 384, 768],           # 무슨 feature ?
                                                hooks=[2, 5, 8, 11],                    # hooks ?
                                                vit_features=768,                       # embed dim ? yes!
                                                use_readout='project')      # DPT 에서는 cls token = readout token 이라고 부르고 projection으로 cls token 처리 

            
    # JINLOVESPHO sf_selfsup_try1
    elif train_args.model_info == 'sf_selfsup_try1':
        
        v = networks.vit.ViT( image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                              patch_size = 16,
                              num_classes = 1000,
                              dim = 768,
                              depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                              heads = 12,
                              mlp_dim = 3072)
        
        if train_args.pretrained_weight == 'vit_base_384':
            is_well_loaded=v.load_state_dict(torch.load("../pretrained_weights/vit_base_384.pth"))
            print(is_well_loaded)
            
        v.resize_pos_embed(192,640)
        
        # show experiment info in terminal
        print_exp_info(train_args)

        breakpoint()
        model['depth'] = networks.SF_Depth_SelfSup_Try1(    encoder=v,
                                                            max_depth = train_args.max_depth,
                                                            features=[96, 192, 384, 768],           # 무슨 feature ?
                                                            hooks=[2, 5, 8, 11],                    # hooks ?
                                                            vit_features=768,                       # embed dim ? yes!
                                                            use_readout='project')
        
        # load monodepth2 pose network
        model["pose_encoder"] = networks.monodepth2_networks.ResnetEncoder(18,True,num_input_images=2 )
        model["pose_decoder"] = networks.monodepth2_networks.PoseDecoder(   model["pose_encoder"].num_ch_enc,
                                                                            num_input_features=1,
                                                                            num_frames_to_predict_for=2)
        
        if train_args.eval:
            pass
            
        
    # JINLOVESPHO sf_selfsup_try2 - change encoder to resnet
    elif train_args.model_info == 'sf_selfsup_try2':
        
        v = networks.monodepth2_networks.resnet_encoder.ResnetEncoder(num_layers=152,  # param 60M
                                                                      pretrained=True,
                                                                      num_input_images=1)
        
        # show experiment info in terminal
        print_exp_info(train_args)

        breakpoint()
        model['depth'] = networks.SF_Depth_SelfSup_Try2(    encoder=v,
                                                            max_depth = train_args.max_depth,
                                                            features=[96, 192, 384, 768],           # 무슨 feature ?
                                                            hooks=[2, 5, 8, 11],                    # hooks ?
                                                            vit_features=768,                       # embed dim ? yes!
                                                            use_readout='project')
        
        
        
        # load monodepth2 pose network
        model["pose_encoder"] = networks.monodepth2_networks.ResnetEncoder(18,True,num_input_images=2 )
        model["pose_decoder"] = networks.monodepth2_networks.PoseDecoder(   model["pose_encoder"].num_ch_enc,
                                                                            num_input_features=1,
                                                                            num_frames_to_predict_for=2)
        
        if train_args.eval:
            pass
            

    
    # JINLOVESPHO mf_sup_baseline
    elif train_args.model_info == 'mf_sup_baseline':
        
        if train_args.vit_type == 'vit_base':
            print('ENCODER: vit_base')
            enc_layers=12
            enc_hidden_dim=768
            enc_mlp_dim=3072
            enc_heads=12
        
        elif train_args.vit_type == 'vit_large':
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
                                        num_prev_frame=train_args.num_prev_frame,
                                        croco = (train_args.pretrained_weight == 'croco'))
        
        if train_args.pretrained_weight == 'croco':
            
            if train_args.vit_type == 'vit_base':
                croco_weight = torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth', map_location=device)
            elif train_args.vit_type == 'vit_large':
                croco_weight = torch.load('./CroCo_V2_ViTLarge_BaseDecoder.pth', map_location=device)

            loaded_weight = {}
            
            for key, value in v.state_dict().items():
                if 'transformer' in key:
                    if '0.norm' in key:
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

        # show experiment info in terminal
        print_exp_info(train_args)
        
        breakpoint()
        model['depth'] = networks.MF_Sup_Baseline(encoder=v,
                                                    max_depth = train_args.max_depth,
                                                    features=[96, 192, 384, 768],           # 무슨 feature ?
                                                    hooks=[2, 5, 8, 11],                    # hooks ?
                                                    vit_features=enc_hidden_dim,                       # embed dim ? yes!
                                                    use_readout='project',
                                                    start_index=1,
                                                    masking_ratio=train_args.masking_ratio,
                                                    cross_attn_depth = train_args.cross_attn_depth,
                                                    croco = (train_args.pretrained_weight == 'croco'),
                                                    )        
    
    # JINLOVESPHO mf_sup_croco_baseline
    elif train_args.model_info == 'mf_sup_croco_baseline':
        from networks.croco_models.croco_downstream import CroCoDownstreamBinocular
        from networks.croco_models.head_downstream import PixelwiseTaskWithDPT
        from networks.croco_models.pos_embed import interpolate_pos_embed
        from networks.mf_sup_croco_baseline import MF_Sup_Croco_Baseline
        
        if train_args.vit_type == 'vit_base':
            print('ENCODER: vit_base')
            enc_layers=12
            enc_hidden_dim=768
            enc_mlp_dim=3072
            enc_heads=12
        
        elif train_args.vit_type == 'vit_large':
            print('ENCODER: vit_large')
            enc_layers=24
            enc_hidden_dim=1024
            enc_mlp_dim=4096
            enc_heads=16
        
        else:
            print('vit type not valid')
            
        
        pw1= torch.load('../pretrained_weights/crocostereo.pth')
        pw2= torch.load('../pretrained_weights/CroCo_V2_ViTBase_SmallDecoder.pth')
        pw3= torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth')
        pw4= torch.load('../pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth')    
        
        ckpt = torch.load(train_args.pretrained_weight_path)
        croco_args = ckpt['args'].croco_args
        croco_args['img_size'] = (train_args.re_height, train_args.re_width)
        
        croco_head = PixelwiseTaskWithDPT()
        croco_head.num_channels = 2     # 1 for only prediction, 2 for pred+confidence
        
        croco_model = CroCoDownstreamBinocular(head=croco_head, **croco_args)
        interpolate_pos_embed(croco_model, ckpt['model'])
        msg = croco_model.load_state_dict(ckpt['model'], strict=True)
        print('CROCO_WEIGHT_WELL_LOADED: ', msg)
        print(croco_args)
        
        # show experiment info in terminal
        train_args=vars(train_args)     # vars ! change ~ to dict type
        train_args.update(croco_args)
        train_args = argparse.Namespace(**train_args)   # dict back to namespace
        print_exp_info(train_args)
        
        breakpoint()
        model['depth'] = MF_Sup_Croco_Baseline( model=croco_model,
                                                )      
        

    # JINLOVESPHO Self Sup Try7
    elif train_args.model_info == 'mf_try7':
        
        if train_args.vit_type == 'vit_base':
            print('ENCODER: vit_base')
            enc_layers=12
            enc_hidden_dim=768
            enc_mlp_dim=3072
            enc_heads=12
        
        elif train_args.vit_type == 'vit_large':
            print('ENCODER: vit_large')
            enc_layers=24
            enc_hidden_dim=1024
            enc_mlp_dim=4096
            enc_heads=16
        
        else:
            print('vit type not valid')

        v = networks.ViT_Multiframe( image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                            patch_size = 16,
                            num_classes = 1000,
                            dim = enc_hidden_dim,
                            depth = enc_layers,                     # transformer 의 layer(attention+ff) 개수 의미
                            heads = enc_heads,
                            mlp_dim = enc_mlp_dim,
                            num_prev_frame=train_args.num_prev_frame,
                            croco = (train_args.pretrained_weight == 'croco'))
        
        if train_args.pretrained_weight == 'croco':

            if train_args.vit_type == 'vit_base':
                croco_weight = torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth', map_location=device)
            elif train_args.vit_type == 'vit_large':
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
        
        # show experiment info in terminal
        print_exp_info(train_args)

        breakpoint()
        model['depth'] = networks.MF_Depth_Try7(encoder=v,
                                                max_depth = train_args.max_depth,
                                                features=[96, 192, 384, 768],           # 무슨 feature ?
                                                hooks=[2, 5, 8, 11],                    # hooks ?
                                                vit_features=enc_hidden_dim,                       # embed dim ? yes!
                                                use_readout='project',
                                                masking_ratio=train_args.masking_ratio,
                                                cross_attn_depth = train_args.cross_attn_depth,
                                                croco = (train_args.pretrained_weight == 'croco'),
                                                )
        
    # JINLOVESPHO Try9 cats
    elif train_args.model_info == 'mf_try9':
        
        if train_args.vit_type == 'vit_base':
            print('ENCODER: vit_base')
            enc_layers=12
            enc_hidden_dim=768
            enc_mlp_dim=3072
            enc_heads=12
        
        elif train_args.vit_type == 'vit_large':
            print('ENCODER: vit_large')
            enc_layers=24
            enc_hidden_dim=1024
            enc_mlp_dim=4096
            enc_heads=16
        
        else:
            print('vit type not valid')

        v = networks.ViT_Multiframe( image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                            patch_size = 16,
                            num_classes = 1000,
                            dim = enc_hidden_dim,
                            depth = enc_layers,                     # transformer 의 layer(attention+ff) 개수 의미
                            heads = enc_heads,
                            mlp_dim = enc_mlp_dim,
                            num_prev_frame=train_args.num_prev_frame,
                            croco = (train_args.pretrained_weight == 'croco'))
        
        if train_args.pretrained_weight == 'croco':

            if train_args.vit_type == 'vit_base':
                croco_weight = torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth', map_location=device)
            elif train_args.vit_type == 'vit_large':
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
        
        # show experiment info in terminal
        print_exp_info(train_args)

        breakpoint()
        model['depth'] = networks.MF_Depth_Try7(encoder=v,
                                                max_depth = train_args.max_depth,
                                                features=[96, 192, 384, 768],           # 무슨 feature ?
                                                hooks=[2, 5, 8, 11],                    # hooks ?
                                                vit_features=enc_hidden_dim,                       # embed dim ? yes!
                                                use_readout='project',
                                                masking_ratio=train_args.masking_ratio,
                                                cross_attn_depth = train_args.cross_attn_depth,
                                                croco = (train_args.pretrained_weight == 'croco'),
                                                )
        
 
    else:
        pass
    

    for key, val in model.items():
        model[key] = nn.DataParallel(val)
        model[key].to(device)
        model[key].train()
        parameters_to_train += list(val.parameters())

    return model, parameters_to_train


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