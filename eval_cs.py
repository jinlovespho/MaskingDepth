import argparse
import os
import torch
from tqdm import tqdm
import wandb

import initialize
import utils
import loss
from eval import visualize, eval_metric, get_eval_dict, visualize_cs

from torchvision.utils import save_image
from networks.monodepth2_networks import compute_depth_losses
import numpy as np
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil 
import cv2 


TRAIN = 0
EVAL  = 1


def get_train_args():
    parser = argparse.ArgumentParser(description='args')
    # Data args 
    parser.add_argument('--data_path',      type=str,   default='/path/to/data')
    parser.add_argument("--dataset",        type=str,   choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", 'kitti_depth_multiframe', 'cityscapes'])
    parser.add_argument("--splits",         type=str,   choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "eigen_temp", 'cityscapes'])
    parser.add_argument('--img_ext',        type=str)
    parser.add_argument('--re_height',      type=int,   default=192)
    parser.add_argument('--re_width',       type=int,   default=640)   
    # Eval args 
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--ckpt_name', type=str)
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--num_workers',    type=int) 
    parser.add_argument('--seed',           type=int)
    # Depth args 
    parser.add_argument('--min_depth',      type=float,     default=0.1)
    parser.add_argument('--max_depth',      type=float,     default=80.0)
    # Loss args
    parser.add_argument('--training_loss', type=str)
    parser.add_argument('--use_future_frame', action='store_true')
    parser.add_argument("--smooth_weight", type=float, default=1e-3)
    # Model args 
    parser.add_argument('--model_info', type=str)
    parser.add_argument('--pretrained_path', type=str)
    parser.add_argument('--attn_agg', action="store_true")
    parser.add_argument('--softmax_attn', action="store_true")
    parser.add_argument('--with_pose', action="store_true")
    parser.add_argument('--encoder_freeze', action='store_true')
    parser.add_argument('--decoder_freeze', action='store_true')
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--single', action="store_true")
    parser.add_argument('--zero_aug', type=float, default=0.0)
    parser.add_argument('--attn_conv4d', action="store_true")
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--moving_masking', type=str, default='None', choices=['no_grad','no_grad_distill','None','no_grad_topk'])
    parser.add_argument('--masking_threshold', type=float, default=0.1)
    parser.add_argument('--attn_agg_tf', action="store_true")
    
    parser.add_argument('--num_prev_frame',         type=int)
    parser.add_argument('--cross_attn_depth',       type=int)
    parser.add_argument('--masking_ratio',          type=float)

    # Logging args 
    parser.add_argument('--log_tool',         type=str)
    parser.add_argument('--wandb_proj_name',  type=str)
    parser.add_argument('--wandb_exp_name',   type=str)
    parser.add_argument('--log_path',         type=str,     default='./path/to/log')
    
    parser.add_argument('--cs_val_path', type=str)
    parser.add_argument('--cs_gt_path', type=str)
    
    args = parser.parse_args()
    return args


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



if __name__ == "__main__":
    
    # get all training args
    train_args = get_train_args()
    train_args.frame_ids=[0,-1,1]
    train_args.scales=[0,1,2,3]
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # set seed
    initialize.seed_everything(train_args.seed)

    # load model
    model, _ = initialize.model_load(train_args, device)

    # load data
    _, val_ds, _, val_loader = initialize.data_loader(train_args, train_args.batch_size, train_args.num_workers)

    # set wandb
    if train_args.log_tool == 'wandb':
        wandb.init( project = train_args.wandb_proj_name,
                    name = train_args.wandb_exp_name,
                    config = train_args,
                    dir=train_args.log_path)

    # CITYSCAPE VIS INDEX
    # val_tot_sample = len(val_loader.dataset)
    # rnd_idx = torch.rand(val_tot_sample).argsort()
    # val_vis_sample = 8 if train_args.batch_size > 8 else train_args.batch_size   
    # vis_rnd_idx = rnd_idx[:val_vis_sample]
    # vis_rnd_idx = vis_rnd_idx.tolist()
    
    
    ckpt_path = train_args.ckpt_path
    if train_args.ckpt_name == 'cs_ours':
        msg1 = model['depth'].load_state_dict(torch.load(f'{ckpt_path}/depth.pth'))
        msg2 = model['pose_encoder'].load_state_dict(torch.load(f'{ckpt_path}/pose_encoder.pth'))
        msg3 = model['pose_decoder'].load_state_dict(torch.load(f'{ckpt_path}/pose_decoder.pth'))
        print('LOADING CKPT CITYSCAPES WEIGHTS')
        print(msg1)
        print(msg2)
        print(msg3)
    elif train_args.ckpt_name == 'cs_manydepth':
        print('already implemented in initialize load_model')
        print('DECIDED not to implement in our framework')
    
    elif train_args.ckpt_name == 'cs_dynamicdepth':
        pass
    
    # load_weight_depth = torch.load('/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu0_kitti_bs16_sf_selfsup_try1_eigenzhou/weights_10/depth.pth')
    # load_weight_pose_enc = torch.load('/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu0_kitti_bs16_sf_selfsup_try1_eigenzhou/weights_10/pose_encoder.pth')
    # load_weight_pose_dec = torch.load('/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu0_kitti_bs16_sf_selfsup_try1_eigenzhou/weights_10/pose_decoder.pth')

    # is_load_depth = model['depth'].module.load_state_dict(load_weight_depth)
    # is_load_pose_enc = model['pose_encoder'].module.load_state_dict(load_weight_pose_enc)
    # is_load_pose_dec = model['pose_decoder'].module.load_state_dict(load_weight_pose_dec)
    # print(is_load_depth)
    # print(is_load_pose_enc)
    # print(is_load_pose_dec)
    
    epoch=0
    # validation
    with torch.no_grad():
        utils.model_mode(model,EVAL)
        eval_loss = 0
        eval_error = []
        pred_depths = []
        gt_depths = []
        
        inputs_color=[]
        mving_msks=[]
        
        pred_depth_npy=[]
        
        # val loop
        tqdm_val = tqdm(val_loader, desc=f'Validation Epoch: {epoch+1}/1')
        for i, inputs in enumerate(tqdm_val):
            # print(torch.cuda.memory_allocated()/1e9)    # for GPU mem tracking
            total_loss = 0
            losses = {}
            
            # move tensors to cuda
            for key, val in inputs.items():
                if type(val) == torch.Tensor:   # not all inputs are tensors
                    inputs[key] = val.to(device)

            # val forward pass
            total_loss, losses, pred_depth_orig, model_outs = loss.compute_loss(inputs, model, train_args, EVAL, epoch=epoch)
            eval_loss += total_loss
            
            if train_args.dataset == 'cityscapes':
                pred_depth_npy.extend(model_outs['pred_depth',0,0].detach().cpu().numpy())  # 192 512
                pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu())
                inputs_color.extend(inputs['color',0,0].detach().cpu())
                mving_msks.extend(inputs['doj_mask'].squeeze(1).detach().cpu())
            
            else:
                gt_depth = inputs['depth_gt']
                gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
                pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu().numpy())
        
        pred_depth_npy = np.concatenate(pred_depth_npy) # num_pred, 768, 2048
        np.save('./ours_pred_depth_192_512.npy', pred_depth_npy)
        
        # breakpoint()
        if train_args.dataset == 'cityscapes':
            MIN_DEPTH = 1e-3
            MAX_DEPTH = 80
            gt_path = train_args.cs_gt_path
            num_gt_samples = len(val_loader.dataset)
            
            all_errors = []     # dynamic and static, normal
            all_ratios = []
            
            dynamic_errors=[]
            dynamic_ratios=[]
            
            static_errors=[]
            static_ratios=[]
            
            vis_gt_depths=[]
            vis_pred_depths=[]
            vis_inputs=[]
            vis_mving_msks=[]
            
            cs_eval_tqdm = tqdm(range(len(pred_depths)), desc=f'CityScapes Eval Epoch: {epoch+1}/1')
            for i in cs_eval_tqdm:
                gt_depth = np.load(os.path.join(gt_path, str(i).zfill(3) + '_depth.npy'))   # 1024 2048
                gt_height, gt_width = gt_depth.shape[:2]    # 1024 2048
                # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
                gt_height = int(round(gt_height * 0.75))    # 768
                gt_depth = torch.from_numpy(gt_depth[:gt_height])    # 768, 2048
                pred_depth = pred_depths[i] # 768, 2048
                
                cs_vis_log_path="/media/dataset1/jinlovespho/aaai_log/SUPPL/cs_vis/ours"
                # pred full size
                vis_pred_depth = pred_depth.cpu().numpy()   # 768 2048
                vmax = np.percentile(vis_pred_depth, 95)
                normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
                cv2.imwrite(f'{cs_vis_log_path}/full/{i}_depth_pred_full.jpg', colormapped_pred_depth[:,:,[2,1,0]])
                # cv2.imwrite(f'./depth_pred_full.jpg', colormapped_pred_depth[:,:,[2,1,0]])
                
                vis_input = inputs_color[i].unsqueeze(dim=0)    # 3 128 416 or 3 192 512
                vis_input = F.interpolate(vis_input, (gt_height, gt_width), mode='bilinear', align_corners=True)      
                vis_input = vis_input.squeeze(dim=0)    # 3 768 2048
                save_image(vis_input, f'{cs_vis_log_path}/rgb/{i}_color.jpg', normalize=True)
                
                mving_msk = mving_msks[i].unsqueeze(dim=0).unsqueeze(dim=0) 
                mving_msk = F.interpolate(mving_msk, (gt_height, gt_width), mode='bilinear', align_corners=True)
                mving_msk = mving_msk.squeeze() # 768 2048
                
                # when evaluating cityscapes, we centre crop to the middle 50% of the image.
                # Bottom 25% has already been removed - so crop the sides and the top here
                gt_depth = gt_depth[256:, 192:1856] # 512 1664
                pred_depth = pred_depth[256:, 192:1856] # 512 1664
                vis_input = vis_input[:, 256:, 192:1856]    # 3 512 1664
                mving_msk = mving_msk[256:, 192:1856]   # 512 1664
                
                # crop
                vis_pred_depth = pred_depth.cpu().numpy()   # 768 2048
                vmax = np.percentile(vis_pred_depth, 95)
                normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
                mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
                cv2.imwrite(f'{cs_vis_log_path}/crop/{i}_depth_pred_crop.jpg', colormapped_pred_depth[:,:,[2,1,0]])
                
                
                # if i in vis_rnd_idx:
                #     # breakpoint()
                #     # print(vis_rnd_idx)
                #     # print(i)
                #     vis_gt_depths.append(gt_depth)
                #     vis_pred_depths.append(pred_depth)
                #     vis_inputs.append(vis_input)
                #     vis_mving_msks.append(mving_msk)

                mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)  # 512, 1664
                dynamic_msk = mask & mving_msk.bool() 
                static_msk = mask & ~mving_msk.bool()
                
                all_pred_depth = pred_depth[mask]
                all_gt_depth = gt_depth[mask]
                all_ratio = torch.median(all_gt_depth) / torch.median(all_pred_depth)
                all_ratios.append(all_ratio)
                all_pred_depth *= all_ratio  
                all_pred_depth = torch.clamp(all_pred_depth, MIN_DEPTH, MAX_DEPTH)
                all_errors.append(compute_depth_errors(all_gt_depth, all_pred_depth)) 
                
                dynamic_pred_depth = pred_depth[dynamic_msk]
                dynamic_gt_depth = gt_depth[dynamic_msk]
                dynamic_ratio = torch.median(dynamic_gt_depth) / torch.median(dynamic_pred_depth)
                dynamic_ratios.append(dynamic_ratio)
                # dynamic_pred_depth *= dynamic_ratio  
                dynamic_pred_depth *= all_ratio     # ALL_RATIO
                dynamic_pred_depth = torch.clamp(dynamic_pred_depth, MIN_DEPTH, MAX_DEPTH)
                dynamic_error = compute_depth_errors(dynamic_gt_depth, dynamic_pred_depth)
                # handle images with no dynamic objects
                if True not in torch.isnan(torch.tensor(dynamic_error)):
                    dynamic_errors.append(dynamic_error)
                                    
                static_pred_depth = pred_depth[static_msk]
                static_gt_depth = gt_depth[static_msk]
                static_ratio = torch.median(static_gt_depth) / torch.median(static_pred_depth)
                static_ratios.append(static_ratio)
                # static_pred_depth *= static_ratio 
                static_pred_depth *= all_ratio      # ALL_RATIO
                static_pred_depth = torch.clamp(static_pred_depth, MIN_DEPTH, MAX_DEPTH)
                static_errors.append(compute_depth_errors(static_gt_depth, static_pred_depth))
                                
                
            all_ratios = torch.tensor(all_ratios)
            all_med = torch.median(all_ratios)
            all_std = torch.std(all_ratios / all_med)
            print(" Scaling all_ratios | all_med: {:0.3f} | all_std: {:0.3f}".format(all_med, all_std))
            
            dynamic_ratios = torch.tensor(dynamic_ratios)
            dynamic_med = torch.median(dynamic_ratios)
            dynamic_std = torch.std(dynamic_ratios / dynamic_med)
            print(" Scaling dynamic_ratios | dynamic_med: {:0.3f} | dynamic_std: {:0.3f}".format(dynamic_med, dynamic_std)) 
            
            static_ratios = torch.tensor(static_ratios)
            static_med = torch.median(static_ratios)
            static_std = torch.std(static_ratios / static_med)  
            print(" Scaling static_ratios | static_med: {:0.3f} | static_std: {:0.3f}".format(static_med, static_std))
            
            all_mean_errors = torch.tensor(all_errors).mean(0)
            dynamic_mean_errors = torch.tensor(dynamic_errors).mean(0)
            static_mean_errors = torch.tensor(static_errors).mean(0)

            print(("{:>8} | " * 7).format("all_abs_rel", "all_sq_rel", "all_rmse", "all_rmse_log", "all_a1", "all_a2", "all_a3"))
            print(("{: 8.3f} | " * 7 + "\n").format(*all_mean_errors.tolist()))  
            
            print(("{:>8} | " * 7).format("dyn_abs_rel", "dyn_sq_rel", "dyn_rmse", "dyn_rmse_log", "dyn_a1", "dyn_a2", "dyn_a3"))
            print(("{: 8.3f} | " * 7 + "\n").format(*dynamic_mean_errors.tolist()))  
            
            print(("{:>8} | " * 7).format("stat_abs_rel", "stat_sq_rel", "stat_rmse", "stat_rmse_log", "stat_a1", "stat_a2", "stat_a3"))
            print(("{: 8.3f} | " * 7 + "\n").format(*static_mean_errors.tolist()))  
            
            depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
            error_dict = {}
            for error_name, error_value in zip(depth_metric_names, all_mean_errors):
                error_dict[error_name] = error_value.item()
            error_dict["val_loss"] = eval_loss / len(val_loader)   
            
            depth_metric_names = ["dynamic_de/abs_rel", "dynamic_de/sq_rel", "dynamic_de/rms", "dynamic_de/log_rms", "dynamic_da/a1", "dynamic_da/a2", "dynamic_da/a3"]
            dyn_error_dict = {}
            for error_name, error_value in zip(depth_metric_names, dynamic_mean_errors):
                dyn_error_dict[error_name] = error_value.item()
            
            depth_metric_names = ["static_de/abs_rel", "static_de/sq_rel", "static_de/rms", "static_de/log_rms", "static_da/a1", "static_da/a2", "static_da/a3"]
            stat_error_dict = {}
            for error_name, error_value in zip(depth_metric_names, static_mean_errors):
                stat_error_dict[error_name] = error_value.item()
                
        else:                
            eval_error = eval_metric(pred_depths, gt_depths, train_args)  
            error_dict = get_eval_dict(eval_error)
            error_dict["val_loss"] = eval_loss / len(val_loader)                

        if train_args.log_tool == 'wandb':
            error_dict["epoch"] = (epoch+1)
            wandb.log(error_dict)
            
            if train_args.dataset == 'cityscapes':
                wandb.log(dyn_error_dict)
                wandb.log(stat_error_dict)
                visualize_cs(vis_inputs, vis_gt_depths, vis_pred_depths, vis_mving_msks, model_outs, train_args)
            else:
                visualize(inputs, pred_depth_orig, model_outs, train_args)
                
    print('End of Epoch')