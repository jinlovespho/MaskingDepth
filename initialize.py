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
from networks.croco.croco_downstream import croco_args_from_ckpt, CroCoDownstreamBinocular
from networks.croco.head_downstream import PixelwiseTaskWithDPT
from networks.croco.pos_embed import interpolate_pos_embed

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

    elif train_args.model_info == 'sf_multiframe_imagenet':
        from networks.masking_depth_dpt import Masked_DPT_Multiframe_Croco_Baseline

        v = networks.ViT_Multiframe( image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                        patch_size = 16,
                        num_classes = 1000,
                        dim = 768,
                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                        heads = 12,
                        mlp_dim = 3072,
                        num_prev_frame=1)
        
        if train_args.pretrained_weight == 'vit_base_384':
            loaded_weight = torch.load("../../seonghoon/MaskingDepth/vit_base_384.pth", map_location=device)
        
            for key, value in v.state_dict().items():
                if key not in loaded_weight.keys():
                    loaded_weight[key] = loaded_weight['pos_embedding']

            is_well_loaded=v.load_state_dict(loaded_weight)
            print(is_well_loaded)

        v.resize_pos_embed(192,640,device)
        model['depth'] = Masked_DPT_Multiframe_Croco_Baseline( encoder=v,
                                                                max_depth = train_args.max_depth,
                                                                features=[96, 192, 384, 768],           # 무슨 feature ?
                                                                hooks=[2, 5, 8, 11],                    # hooks ?
                                                                vit_features=768,                       # embed dim ? yes!
                                                                use_readout='project',
                                                                masking_ratio=0.0)

        model["pose_encoder"] = networks.monodepth2_networks.ResnetEncoder(18,True,num_input_images=2 )
        model["pose_decoder"] = networks.monodepth2_networks.PoseDecoder(   model["pose_encoder"].num_ch_enc,
                                                                                num_input_features=1,
                                                                                num_frames_to_predict_for=2)

        params_to_train+=model['depth'].parameters()
        params_to_train+=model['pose_encoder'].parameters()
        params_to_train+=model['pose_decoder'].parameters()
        
    
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

    elif train_args.model_info == 'croco':
        ckpt = torch.load(train_args.pretrained_path, 'cpu')
        croco_args = croco_args_from_ckpt(ckpt)
        croco_args['img_size'] = (192, 640)
        croco_args['mask_ratio'] = train_args.mask_ratio
        # croco_args['attn_conv4d'] = train_args.attn_conv4d
        croco_args['args'] = train_args
        
        print('Croco args: '+str(croco_args))
        num_channels = 1
        print(f'Building head PixelwiseTaskWithDPT() with {num_channels} channel(s)')
        
        if train_args.attn_agg:
            head = PixelwiseTaskWithDPT(attn_agg = train_args.attn_agg, hooks_idx=[14,17,20,23], with_pose = train_args.with_pose,residual = train_args.residual, single = train_args.single)
        else:
            if train_args.attn_conv4d:
                head = PixelwiseTaskWithDPT(residual=train_args.residual, single=train_args.single, hooks_idx=[14,17,20,23], args=train_args,with_pose=train_args.with_pose)
            else:
                head = PixelwiseTaskWithDPT(residual=train_args.residual, single=train_args.single, args=train_args)
        head.num_channels = num_channels
        model['depth'] = CroCoDownstreamBinocular(head, **croco_args)
        interpolate_pos_embed(model['depth'], ckpt['model'])
        
        msg = model['depth'].load_state_dict(ckpt['model'], strict=False)
        
        if not train_args.with_pose:
            model["pose_encoder"] = networks.monodepth2_networks.ResnetEncoder(18,True,num_input_images=2 )
            model["pose_decoder"] = networks.monodepth2_networks.PoseDecoder(   model["pose_encoder"].num_ch_enc,
                                                                                num_input_features=1,
                                                                                num_frames_to_predict_for=2)
    
    
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
            is_well_loaded=v.load_state_dict(torch.load("../../seonghoon/MaskingDepth/vit_base_384.pth"))
            print(is_well_loaded)
            
        if train_args.pretrained_weight == 'croco':
            if train_args.vit_type == 'vit_base':
                croco_weight = torch.load('../pretrained_weights/CroCo_V2_ViTBase_BaseDecoder.pth', map_location=device)

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
            
        v.resize_pos_embed(192,640)
        
        # show experiment info in terminal
        print_exp_info(train_args)

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
        
        params_to_train+=model['depth'].parameters()
        params_to_train+=model['pose_encoder'].parameters()
        params_to_train+=model['pose_decoder'].parameters()
        
        # model_params = [param for name, param in model['depth'].model.named_parameters()]
        # backbone_params = [param for name, param in model['depth'].model.named_parameters() if 'enc_block' in name]
        # else_params = [param for name, param in model['depth'].model.named_parameters() if 'enc_block' not in name]
        
        # params_to_train.append( {'params':backbone_params, 'lr':train_args.backbone_lr} )
        # params_to_train.append( {'params':else_params, 'lr':train_args.lr})
        
        
        # # validation
        # ckpt_path=f'/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu1_kitti_bs16_sf_selfsup_try1_eigenzhou_re_depth_metric_maxdepth80_cornersTrue/weights_20'
        # depth_weight=f'{ckpt_path}/depth.pth'
        # pose_enc_weight=f'{ckpt_path}/pose_encoder.pth'
        # pose_dec_weight=f'{ckpt_path}/pose_decoder.pth'
        
        # depth_weight = torch.load(depth_weight)
        # pose_enc_weight=torch.load(pose_enc_weight)
        # pose_dec_weight=torch.load(pose_dec_weight)
        
        # msg1=model['depth'].load_state_dict(depth_weight, strict=True)
        # msg2=model['pose_encoder'].load_state_dict(pose_enc_weight, strict=True)
        # msg3=model['pose_decoder'].load_state_dict(pose_dec_weight, strict=True)
        # print(msg1,msg2,msg3)
        
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
            

        
        
 
    else:
        pass
    
    if train_args.load_weight_path is not None:
        print('load_weight_path')
        model['depth'].load_state_dict(torch.load(os.path.join(train_args.load_weight_path,'depth.pth')))
        model['pose_encoder'].load_state_dict(torch.load(os.path.join(train_args.load_weight_path,'pose_encoder.pth')))
        model['pose_decoder'].load_state_dict(torch.load(os.path.join(train_args.load_weight_path,'pose_decoder.pth')))
    

    for key, val in model.items():
        model[key] = nn.DataParallel(val)
        model[key].to(device)
        model[key].train()
        # parameters_to_train += list(val.parameters())

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