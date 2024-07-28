import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
from PIL import Image 

import initialize
import utils
import loss
from vis_eval import visualize, eval_metric, get_eval_dict

from torchvision.utils import save_image
from networks.monodepth2_networks import compute_depth_losses

import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import matplotlib.pyplot as plt


TRAIN = 0
EVAL  = 1


def get_train_args():
    parser = argparse.ArgumentParser(description='args')
    # Data args 
    parser.add_argument('--data_path',      type=str,   default='/path/to/data')
    parser.add_argument("--dataset",        type=str,   choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", 'kitti_depth_multiframe'])
    parser.add_argument("--splits",         type=str,   choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "eigen_temp"])
    parser.add_argument('--img_ext',        type=str)
    parser.add_argument('--re_height',      type=int,   default=192)
    parser.add_argument('--re_width',       type=int,   default=640)   
    # Val args 
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--num_workers',    type=int) 
    parser.add_argument('--seed',           type=int)
    # Depth args 
    parser.add_argument('--min_depth',      type=float,     default=0.1)
    parser.add_argument('--max_depth',      type=float,     default=80.0)
    # Loss args
    parser.add_argument('--training_loss',  type=str)
    parser.add_argument('--use_future_frame',   action='store_true')
    parser.add_argument("--smooth_weight", type=float, default=1e-3)
    # Model args 
    parser.add_argument('--model_info',             type=str)
    parser.add_argument('--vit_type',               type=str,   default='vit_base')
    parser.add_argument('--pretrained_weight',      type=str)
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--attn_agg', action="store_true")
    parser.add_argument('--softmax_attn', action="store_true")
    parser.add_argument('--with_pose', action="store_true")
    parser.add_argument('--encoder_freeze', action='store_true')
    parser.add_argument('--decoder_freeze', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--single', action="store_true")
    parser.add_argument('--zero_aug', type=float, default=0.0)
    parser.add_argument('--load_weight_path',       type=str, default=None)
    # Save args 
    parser.add_argument("--epoch_save_freq", type=int, default=5)
    # Logging args 
    parser.add_argument('--log_tool',         type=str)
    parser.add_argument('--wandb_proj_name',  type=str)
    parser.add_argument('--wandb_exp_name',   type=str)
    parser.add_argument('--log_path',         type=str,     default='./path/to/log')
    # Etc args
    parser.add_argument('--eval', action='store_true')
    
    args = parser.parse_args()
    return args

def show_mask_on_image(img, mask):
    img = np.float32(img)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == "__main__":
    
    # get all training args
    train_args = get_train_args()
    train_args.frame_ids=[0,-1,1]
    train_args.scales=[0,1,2,3]
     
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set seed
    initialize.seed_everything(train_args.seed)

    # model_load
    model, _ = initialize.model_load(train_args, device)
    
    # data loader
    train_ds, val_ds, train_loader, val_loader = initialize.data_loader(train_args, train_args.batch_size, train_args.num_workers)
                                            
    # set wandb
    if train_args.log_tool == 'wandb':
        wandb.init( project = train_args.wandb_proj_name,
                    name = train_args.wandb_exp_name,
                    config = train_args,
                    dir=train_args.log_path)

    # set hooks to visualize cross attention maps in decoder
    ca1_names=[]
    ca1_modules=[]
    ca1_maps=[]
    for name, module in model['depth'].named_modules():
        if 'dec_blocks' in name and 'cross_attn.attn_drop' in name:
            ca1_names.append(name)
            ca1_modules.append(module)
            module.register_forward_hook(lambda m,i,o: ca1_maps.append(o.detach().cpu()) )

    # validation
    with torch.no_grad():
        utils.model_mode(model,EVAL)
        eval_loss = 0
        eval_error = []
        pred_depths = []
        gt_depths = []

        # val loop
        tqdm_val = tqdm(val_loader, desc=f'Validation Epoch: Only Once')
        for k, inputs in enumerate(tqdm_val):
            
            total_loss = 0
            losses = {}
            
            # move tensors to cuda
            for key, val in inputs.items():
                if type(val) == torch.Tensor:   # not all inputs are tensors
                    inputs[key] = val.to(device)
        
            # val forward pass
            total_loss, losses, pred_depth_orig, model_outs = loss.compute_loss(inputs, model, train_args, EVAL)
            
            # attn map visualize
            img_curr = inputs['color',0,0]      # b 3 192 640
            img_prev = inputs['color',-1,0]
            
            ca1_map = torch.stack(ca1_maps, dim=1)  # b num_layer num_head N1 N2
            
            ca1_maps.clear()
            
            ca1_map = ca1_map.mean(dim=2)   # b num_layer N1 N2
            
            log_name=f'.vis_attn/{train_args.model_info}/iter{k}'
            if not os.path.exists(log_name):
                os.makedirs(log_name)
                
            vis_num_points=20
            rnd = torch.rand(480).argsort()
            
            for tkn_vis_idx in rnd[:vis_num_points]:
                tkn_vis_idx=tkn_vis_idx.item()
                vis_idx_h = int(tkn_vis_idx//40 * 16)
                vis_idx_w = int(tkn_vis_idx%40 * 16)
                pt1 = (vis_idx_w, vis_idx_h)
                pt2 = (vis_idx_w+16, vis_idx_h+16)
        
                for i in range(train_args.batch_size):  # batch별
                    
                    i_img_curr=img_curr[i].detach().cpu()  # 3 192 640
                    i_img_prev=img_prev[i].detach().cpu()
                    i_ca1_map=ca1_map[i].detach().cpu()    # num_layer N1 N2
                    
                    vis_pred_depth = pred_depth_orig[i].squeeze().cpu().numpy()     # 1 375 1242
                    vmax = np.percentile(vis_pred_depth, 95)
                    normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
                    colormapped_pred_depth = cv2.resize(colormapped_pred_depth, (640,192))
                    vis_pred_depth = Image.fromarray(colormapped_pred_depth)
                      
                    # save_image(i_img_curr, f'{log_name}/{i}_img_curr.jpg')
                    # save_image(i_img_prev, f'{log_name}/{i}_img_prev.jpg')
                    
                    # log current image and query point
                    i_img_curr_np = np.float32(i_img_curr.permute(1,2,0)) * 255 
                    i_img_curr_np = i_img_curr_np[:,:,[2,1,0]]
                    i_img_curr_query=cv2.rectangle(i_img_curr_np.astype(np.uint8).copy(), pt1, pt2, (0,0,255), -1)
                    log_name2=f'{log_name}/tkn{tkn_vis_idx}_h{vis_idx_h}_w{vis_idx_w}'
                    if not os.path.exists(log_name2):
                        os.makedirs(log_name2)
                    file_name=f'batch{i}_img_curr.jpg'
                    cv2.imwrite(f'{log_name2}/aaa_tmp_curr{i}.jpg', i_img_curr_query) 
                    cv2.imwrite(f'{log_name2}/{file_name}', i_img_curr_query)  
                    vis_pred_depth.save(f'{log_name2}/{file_name}_pred_depth.jpg')
                    i_ca1_map[:,:,441] = 0
                    for j in range(12):     # layer별
                        j_ca1_map=i_ca1_map[j]  # N1 N2 
                        
                        
                        vis_tkn_ca1 = j_ca1_map[tkn_vis_idx]
                        

                        interp_vis_tkn_ca1 = F.interpolate(vis_tkn_ca1.view(12,40).unsqueeze(dim=0).unsqueeze(dim=0), size=(192,640), mode='bilinear', align_corners=True)
                        
                        # normalize attn map weight to [0,1]
                        interp_vis_tkn_ca1 = (interp_vis_tkn_ca1 - interp_vis_tkn_ca1.min()) / (interp_vis_tkn_ca1.max() - interp_vis_tkn_ca1.min())
                        
                        # show attention map on previous image
                        result_ca = show_mask_on_image(i_img_prev.permute(1,2,0), interp_vis_tkn_ca1.squeeze())    # 192 640 3
                        
                        # cv2.imwrite(f'{log_name}/iter{k}_batch{i}_h{vis_idx_h}_w{vis_idx_w}_img_prev_SA_layer{j}.jpg', result_sa)
                        cv2.imwrite(f'{log_name2}/batch{i}_img_prev_CA_layer{j}.jpg', result_ca)
                        
                        # save_image(interp_vis_tkn_sa1.squeeze(), f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_interp_sa_map_n.jpg', normalize=True)
                        # save_image(interp_vis_tkn_ca1.squeeze(), f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_interp_ca_map_n.jpg', normalize=True)
                        
                        # save_image(vis_tkn_sa1, f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_sa_map_n.jpg', normalize=True)
                        # save_image(vis_tkn_ca1, f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_ca_map_n.jpg', normalize=True)
                    
                #     print('visualized per layer')
                # print('visualized per point')
            # print('visualized')

            eval_loss += total_loss
            
            gt_depth = inputs['depth_gt']
            pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu().numpy())
            gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
        
        eval_error = eval_metric(pred_depths, gt_depths, train_args)  
        error_dict = get_eval_dict(eval_error)
        error_dict["val_loss"] = eval_loss / len(val_loader)   
                 

        if train_args.log_tool == 'wandb':
            error_dict["epoch"] = (1)
            wandb.log(error_dict)
            visualize(inputs, pred_depth_orig, model_outs, train_args)
                
    print('End of Epoch')