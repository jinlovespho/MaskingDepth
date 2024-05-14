import os
import argparse

import yaml
from dotmap import DotMap
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb

import initialize
import utils
import loss
from eval import visualize, eval_metric, get_eval_dict

from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
import cv2 

from vit_rollout import rollout

TRAIN = 0
EVAL  = 1

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./conf/base_train.yaml")

args = parser.parse_args()

def return_mask_tensor_numpy(tkn):
    # kitti
    patch_num_h=12
    patch_num_w=40
    resized_h = 192
    resized_w = 640
    orig_h=375
    orig_w=1242
    
    # breakpoint()
    
    tkn = tkn.reshape(patch_num_h, patch_num_w).numpy()
    tkn = tkn / tkn.max()
    mask = cv2.resize(tkn, (resized_w, resized_h) )
    
    return mask
 
    tkn = tkn.view(patch_num_h, patch_num_w)    # (12,40)
    
    tkn = tkn / tkn.max()
    tkn = tkn[None,None,...]
    vis = nn.functional.interpolate(tkn, size=(resized_h, resized_w), mode='bilinear')
    vis_t = (vis-vis.min())/(vis.max()-vis.min())
    
    vis_n = vis_t.numpy() * 255 
    vis_n = np.uint8(vis_n)
    
    return vis_t, vis_n
       
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def pho_visualize_cross_attn_map(ca_module_num, iter_num, curr_frame, prev_frame, ca_maps, discard_ratio, vis_idx):

    curr_frame_n = curr_frame[0,...].detach().cpu().permute(1,2,0).numpy() *255    # (192,640,3) 
    prev_frame_n = prev_frame[0,...].detach().cpu().permute(1,2,0).numpy() *255     # (192,640,3)
        
    curr_frame_n = curr_frame_n[:,:,[2,1,0]]
    prev_frame_n = prev_frame_n[:,:,[2,1,0]]
    
    # ca_map = rollout(ca_maps, discard_ratio=discard_ratio, head_fusion='mean')    # (n,n) = (480,480)
    ca_map = torch.cat(ca_maps,dim=1)
    ca_map = ca_map.mean(dim=(0,1))

    # token to visualize ca_map
    to_vis_tkn = ca_map[vis_idx,:]  # (480)
    h_idx = vis_idx//40
    w_idx = vis_idx%40
    
    # show ca_map
    mask= return_mask_tensor_numpy(to_vis_tkn)  # (192,640)
    result = show_mask_on_image(prev_frame_n, mask) # (192,640,3)
    
    # plot query point for curr_frame
    query=torch.zeros(12,40)
    query[h_idx,w_idx]=1
    query = nn.functional.interpolate(query[None,None,...], size=(192,640), mode='bilinear')
    
    query_numpy = query[0,0,...].detach().cpu().numpy()    # (192,640)

    # show query_point on curr_frame
    query_result = show_mask_on_image(curr_frame_n, query_numpy) # (192,640,3)

    if iter_num == 0 and ca_module_num ==1:
        cv2.imwrite(f'./vis_ca_map/try4_conv/vis_idx{vis_idx}_ca_map_curr.jpg', query_result)
    cv2.imwrite(f'./vis_ca_map/try4_conv/vis_idx{vis_idx}_mod{ca_module_num}_ca_map_prev.jpg', result )
    

if __name__ == "__main__":
    with open(args.conf, 'r') as f:
        conf =  yaml.load(f, Loader=yaml.FullLoader)
        train_cfg = DotMap(conf['Train'])
        device = torch.device("cuda" if train_cfg.use_cuda else "cpu")
        
        # seed 
        initialize.seed_everything(train_cfg.seed)
        
        # JINLOVESPHO
        train_cfg.model.num_prev_frame=train_cfg.data.num_prev_frame

        #model_load
        model, parameters_to_train = initialize.baseline_model_load(train_cfg.model, device)

        #optimizer & scheduler
        encode_index = len(list(model['depth'].module.encoder.parameters()))
        optimizer = optim.Adam([{"params": parameters_to_train[:encode_index], "lr": 1e-5}, 
                                {"params": parameters_to_train[encode_index:]}], float(train_cfg.lr))
        
        if train_cfg.load_optim:
            print('Loading Adam optimizer')
            optim_file = os.path.join(train_cfg.model.weight_path,'adam.pth')
            if os.path.isfile(optim_file):
                print("Success load optimizer")
                optim_load_dict = torch.load(optim_file, map_location=device)
                optimizer.load_state_dict(optim_load_dict)
            else:
                print(f"Dose not exist {optim_file}")
        
        if train_cfg.lr_scheduler:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, train_cfg.scheduler_step_size, 0.1)
            
        # data loader
        train_ds, val_ds, train_loader, val_loader = initialize.data_loader(train_cfg.data, train_cfg.batch_size, train_cfg.num_workers)
                                        
        # set wandb
        if train_cfg.wandb:
            wandb.init(project = train_cfg.wandb_proj_name,
                        name = train_cfg.model_name,
                        config = conf,
                        dir=train_cfg.wandb_log_path)
        # save configuration (this part activated when do not use wandb)
        else: 
            save_config_folder = os.path.join(train_cfg.log_path, train_cfg.model_name)
            if not os.path.exists(save_config_folder):
                os.makedirs(save_config_folder)
            with open(save_config_folder + '/config.yaml', 'w') as f:
                yaml.dump(conf, f)
            progress = open(save_config_folder + '/progress.txt', 'w')
                
        step = 0    
        #validation
        with torch.no_grad():
            utils.model_mode(model,EVAL)
            
            mod = model['depth'].module
            
            # LOAD MODEL and VISUALIZE ATTENTION MAPS
            # load_path = '/media/dataset1/jinlovespho/log/multiframe/pho_gpu2_kitti_bs4_mask00_fuse(try4a_conv4d_sum)/weights_10/depth.pth'
            # loaded_weight = torch.load(load_path)
            # load_result = model['depth'].module.load_state_dict(loaded_weight, strict=False)
            
            eval_loss = 0
            eval_error = []
            pred_depths = []
            gt_depths = []

            # validation loop
            for i, inputs in enumerate(tqdm(val_loader)):
                
                breakpoint()
                
                total_loss = 0
                losses = {}
            
                # multiframe validation
                if train_cfg.data.dataset=='kitti_depth_multiframe':
                    for input in inputs:
                        for key, ipt in input.items():
                            if type(ipt) == torch.Tensor:
                                input[key] = ipt.to(device)     # Place current and previous frames on cuda

                    if train_cfg.model.enable_color_loss:
                        total_loss, _, pred_depth, pred_color, pred_uncert, pred_depth_mask = loss.compute_loss_multiframe_colorLoss(inputs, model, train_cfg, EVAL)    
                    else:
                        total_loss, _, pred_depth, pred_uncert, pred_depth_mask = loss.compute_loss_multiframe(inputs, model, train_cfg, EVAL)   
                                                           
                    gt_depth = inputs[0]['depth_gt']
                    gt_color = inputs[0]['color']
                    
                # singleframe validation
                else:
                    for key, ipt in inputs.items():
                        if type(ipt) == torch.Tensor:
                            inputs[key] = ipt.to(device)
                    # with torch.cuda.amp.autocast(enabled=True):
                    total_loss, _, pred_depth, pred_uncert, pred_depth_mask = loss.compute_loss(inputs, model, train_cfg, EVAL)
                    gt_depth = inputs['depth_gt']
            
                # breakpoint()
                eval_loss += total_loss
                # pred_depth.squeeze(dim=1)은 tensor 로 (8,H,W) 이고. pred_depths 는 [] 리스트이다.
                # pred_depths.extend( pred_depth )를 해주면 pred_depth 의 8개의 이미지들이 차례로 리스트로 들어가서 리스트 len은 개가 돼
                # 즉 list = [ pred_img1(H,W), pred_img2(H,W), . . . ] 
                pred_depths.extend(pred_depth.squeeze(1).cpu().numpy())
                gt_depths.extend(gt_depth.squeeze(1).cpu().numpy())

            # breakpoint()
            eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
            error_dict = get_eval_dict(eval_error)
            error_dict["val_loss"] = eval_loss / len(val_loader)      
            
            print(error_dict)          
