import random
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import datasets
import networks
import utils

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

def baseline_model_load(model_cfg, device):
    model = {}
    parameters_to_train = []

    if model_cfg.baseline == 'DPT':
        v = networks.ViT(image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                        patch_size = 16,
                        num_classes = 1000,
                        dim = 768,
                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                        heads = 12,
                        mlp_dim = 3072)
        v.load_state_dict(torch.load("./pretrained_weights/vit_base_384.pth"))
        v.resize_pos_embed(192,640)

        breakpoint()
        model['depth'] = networks.Masked_DPT(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project')      # DPT 에서는 cls token = readout token 이라고 부르고 projection으로 cls token 처리 
        
    elif model_cfg.baseline == 'DPT_H':
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
                        max_depth = model_cfg.max_depth,
                        use_readout='project')
                                
    elif model_cfg.baseline == 'monodepth2':
        resnet_encoder = networks.ResnetEncoder(50, True, mask_layer=3)
        depth_decoder = networks.DepthDecoder(num_ch_enc=resnet_encoder.num_ch_enc, scales=range(4))
        model['depth'] = networks.Monodepth(resnet_encoder, depth_decoder, max_depth = model_cfg.max_depth)
        
    # JINLOVESPHO        
    elif model_cfg.baseline == 'DPT_Multiframe':
        v = networks.ViT_Multiframe(   image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = 768,
                                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = 12,
                                        mlp_dim = 3072,
                                        num_prev_frame=model_cfg.num_prev_frame)
        
        loaded_weight = torch.load("./pretrained_weights/vit_base_384.pth", map_location=device)
        
        # 현재 v = ViT_Multiframe 에는 positional embedding 을 새로 여러개 만들어줌
        # 하지만 torch.load에서 loaded_weight에는 이 pos emb pretrained 된 weight에 없기에
        # loaded_weight에 새로 추가해주고 이후에 덮어씌우는 것 
        
        for key, value in v.state_dict().items():
            if key not in loaded_weight.keys():
                loaded_weight[key] = loaded_weight['pos_embedding']
        
        v.load_state_dict(loaded_weight)
        # v.load_state_dict(torch.load("./pretrained_weights/vit_base_384.pth"))
        v.resize_pos_embed(192,640,device)     # resize all positional embeddings in ViT_Multiframe class member variables

        model['depth'] = networks.Masked_DPT_Multiframe(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=model_cfg.num_prev_frame)      # DPT 에서는 cls token = readout token 이라고 부르고 projection으로 cls token 처리 
    
    # JINLOVESPHO        
    elif model_cfg.baseline == 'DPT_Multiframe_mask_t':
        v = networks.ViT_Multiframe(   image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = 768,
                                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = 12,
                                        mlp_dim = 3072,
                                        num_prev_frame=model_cfg.num_prev_frame)
        
        loaded_weight = torch.load("./pretrained_weights/vit_base_384.pth", map_location=device)
        
        for key, value in v.state_dict().items():
            if key not in loaded_weight.keys():
                loaded_weight[key] = loaded_weight['pos_embedding']
        
        v.load_state_dict(loaded_weight)
        v.resize_pos_embed(192,640,device)     # resize all positional embeddings in ViT_Multiframe class member variables

        breakpoint()
        model['depth'] = networks.Masked_DPT_Multiframe_mask_t(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=model_cfg.num_prev_frame,
                        masking_ratio=model_cfg.masking_ratio,
                        num_frame_to_mask=model_cfg.num_frame_to_mask)
    
    # JINLOVESPHO -> croco 처럼 cross attention implement  
    elif model_cfg.baseline == 'DPT_Multiframe_CrossAttn_mask_t':
        v = networks.ViT_Multiframe(    image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = 768,
                                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = 12,
                                        mlp_dim = 3072,
                                        num_prev_frame=model_cfg.num_prev_frame)
        
        loaded_weight = torch.load("./pretrained_weights/vit_base_384.pth", map_location=device)
        
        for key, value in v.state_dict().items():
            if key not in loaded_weight.keys():
                loaded_weight[key] = loaded_weight['pos_embedding']
        
        v.load_state_dict(loaded_weight)
        v.resize_pos_embed(192,640,device)     # resize all positional embeddings in ViT_Multiframe class member variables

        breakpoint()
        model['depth'] = networks.Masked_DPT_Multiframe_CrossAttn_mask_t(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=model_cfg.num_prev_frame,
                        masking_ratio=model_cfg.masking_ratio,
                        num_frame_to_mask=model_cfg.num_frame_to_mask,
                        cross_attn_depth = 8
                        )
    
    # JINLOVESPHO
    elif model_cfg.baseline == 'DPT_Multiframe_MultiCrossAttn_mask_t':
        v = networks.ViT_Multiframe(    image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = 768,
                                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = 12,
                                        mlp_dim = 3072,
                                        num_prev_frame=model_cfg.num_prev_frame)
        
        loaded_weight = torch.load("./pretrained_weights/vit_base_384.pth", map_location=device)
        
        for key, value in v.state_dict().items():
            if key not in loaded_weight.keys():
                loaded_weight[key] = loaded_weight['pos_embedding']
        
        v.load_state_dict(loaded_weight)
        v.resize_pos_embed(192,640,device)     # resize all positional embeddings in ViT_Multiframe class member variables

        breakpoint()
        model['depth'] = networks.Masked_DPT_Multiframe_MultiCrossAttn_mask_t(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=model_cfg.num_prev_frame,
                        masking_ratio=model_cfg.masking_ratio,
                        num_frame_to_mask=model_cfg.num_frame_to_mask,
                        cross_attn_depth = model_cfg.cross_attn_depth
                        )
    
    # JINLOVESPHO
    elif model_cfg.baseline == 'DPT_Multiframe_MultiCrossAttn_ColorLoss_mask_t':
        v = networks.ViT_Multiframe(    image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = 768,
                                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = 12,
                                        mlp_dim = 3072,
                                        num_prev_frame=model_cfg.num_prev_frame)
        
        loaded_weight = torch.load("./pretrained_weights/vit_base_384.pth", map_location=device)
        
        for key, value in v.state_dict().items():
            if key not in loaded_weight.keys():
                loaded_weight[key] = loaded_weight['pos_embedding']
        
        v.load_state_dict(loaded_weight)
        v.resize_pos_embed(192,640,device)     # resize all positional embeddings in ViT_Multiframe class member variables

        breakpoint()
        model['depth'] = networks.Masked_DPT_Multiframe_MultiCrossAttn_ColorLoss_mask_t(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=model_cfg.num_prev_frame,
                        masking_ratio=model_cfg.masking_ratio,
                        num_frame_to_mask=model_cfg.num_frame_to_mask,
                        cross_attn_depth = model_cfg.cross_attn_depth
                        )
    
    # JINLOVESPHO
    elif model_cfg.baseline == 'DPT_Multiframe_Croco':
        v = networks.ViT_Multiframe(    image_size = (384,384),        # DPT 의 ViT-Base setting 그대로 가져옴. 
                                        patch_size = 16,
                                        num_classes = 1000,
                                        dim = 768,
                                        depth = 12,                     # transformer 의 layer(attention+ff) 개수 의미
                                        heads = 12,
                                        mlp_dim = 3072,
                                        num_prev_frame=model_cfg.num_prev_frame)
        
        loaded_weight = torch.load("./pretrained_weights/vit_base_384.pth", map_location=device)
        
        for key, value in v.state_dict().items():
            if key not in loaded_weight.keys():
                loaded_weight[key] = loaded_weight['pos_embedding']
        
        v.load_state_dict(loaded_weight)
        v.resize_pos_embed(192,640,device)     # resize all positional embeddings in ViT_Multiframe class member variables

        breakpoint()
        model['depth'] = networks.Masked_DPT_Multiframe_Croco(encoder=v,
                        max_depth = model_cfg.max_depth,
                        features=[96, 192, 384, 768],           # 무슨 feature ?
                        hooks=[2, 5, 8, 11],                    # hooks ?
                        vit_features=768,                       # embed dim ? yes!
                        use_readout='project',
                        num_prev_frame=model_cfg.num_prev_frame,
                        masking_ratio=model_cfg.masking_ratio,
                        num_frame_to_mask=model_cfg.num_frame_to_mask,
                        cross_attn_depth = model_cfg.cross_attn_depth
                        )
       
    else:
        pass
    
    # feature loss
    if model_cfg.mlp_head:
        in_c = []
        out_c = 0
        if model_cfg.baseline == 'DPT':
            in_c = [768,768,768,768]
            out_c = 1024
        elif model_cfg.baseline == 'DPT_H':
            in_c = [768,768,768,768]
            out_c = 1024
        elif model_cfg.baseline == 'monodepth2':
            in_c = [64,256,512,1024,2048]
            out_c = 1280
        else:
            pass

        model['mlp_head'] = networks.MLPHead(in_c, out_c)

    if model_cfg.load_weight:
        print("Loading Network weights")
        depth_file = os.path.join(model_cfg.weight_path, 'depth.pth')
        if os.path.isfile(depth_file):
            print("Success load depth weight")
            model_load_dict = torch.load(depth_file, map_location=device)
            model['depth'].load_state_dict(model_load_dict)
        else:
            print(f"Dose not exist {depth_file}")

        if model_cfg.mlp_head:
            head_file = os.path.join(model_cfg.weight_path, 'mlp_head.pth')
            if os.path.isfile(head_file):
                print("Success load mlp_head weight")
                head_load_dict = torch.load(head_file, map_location=device)
                model['mlp_head'].load_state_dict(head_load_dict)
            else:
                print(f"Dose not exist {head_file}")

    for key, val in model.items():
        model[key] = nn.DataParallel(val)
        model[key].to(device)
        model[key].train()
        parameters_to_train += list(val.parameters())
        
    return model, parameters_to_train

# additional modle load (uncertain network)
def additional_model_load(model_cfg, device):
    model = {}
    parameters_to_train = []
    
    if model_cfg.uncert:
        model['uncert_decoder'] = networks.UncertDecoder()

        if model_cfg.load_weight:
            uncert_file = os.path.join(model_cfg.weight_path, 'uncert_decoder.pth')
            if os.path.isfile(uncert_file):
                uncert_load_dict = torch.load(uncert_file, map_location=device)
                model['uncert_decoder'].load_state_dict(uncert_load_dict)     
    
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
    val_loader = DataLoader(val_dataset, batch_size, True, num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader

def double_data_loader(labeled_cfg, unlabeled_cfg, batch_size, num_workers):
    RATIO = 7
    
    #fix match (semi supervised)
    if labeled_cfg.dataset == unlabeled_cfg.dataset:
        train_loader_label, _   = data_loader(labeled_cfg,    batch_size,          num_workers)
        train_loader_unlabel, _ = data_loader(unlabeled_cfg,  (batch_size * RATIO),   (num_workers*RATIO))
    #pix match (domain adaptation)
    else:
        train_loader_label , _  = data_loader(labeled_cfg,    batch_size//2, num_workers//2)
        train_loader_unlabel, _ = data_loader(unlabeled_cfg,  batch_size//2, num_workers//2)
        
    _, val_loader = data_loader(unlabeled_cfg, batch_size*RATIO, num_workers*RATIO)
    
    return train_loader_label, train_loader_unlabel, val_loader