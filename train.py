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
    # Training args 
    parser.add_argument('--num_epoch',      type=int)  
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--backbone_lr',    type=float)
    parser.add_argument('--learning_rate',             type=float) 
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
    parser.add_argument('--attn_conv4d', action="store_true")
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument('--moving_masking', type=str, default='None', choices=['no_grad','no_grad_distill','None','no_grad_topk'])
    parser.add_argument('--masking_threshold', type=float, default=0.1)
    parser.add_argument('--attn_agg_tf', action="store_true")
    
    parser.add_argument('--num_prev_frame',         type=int)
    parser.add_argument('--cross_attn_depth',       type=int)
    parser.add_argument('--masking_ratio',          type=float)
    # Save args 
    parser.add_argument("--epoch_save_freq", type=int, default=5)
    # Logging args 
    parser.add_argument('--log_tool',         type=str)
    parser.add_argument('--wandb_proj_name',  type=str)
    parser.add_argument('--wandb_exp_name',   type=str)
    parser.add_argument('--log_path',         type=str,     default='./path/to/log')
    # Etc args
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--load_weight_path', type=str, default=None)
    
    
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

    # model_load
    model, params_to_train = initialize.model_load(train_args, device)
    
    #optimizer & scheduler
    if train_args.model_info != 'croco':
        encode_index = len(list(model['depth'].module.encoder.parameters()))
        optimizer = torch.optim.Adam([{"params": params_to_train[:encode_index], "lr": 1e-5}, 
                                    {"params": params_to_train[encode_index:]}  ], float(train_args.learning_rate))
    else:
        pretrained_params, other_params = [], []
        for name, param in model['depth'].named_parameters():
            if 'enc_blocks' in name or 'dec_blocks' in name:
                pretrained_params.append(param)
            # if 'enc_blocks' in name:
                # pretrained_params.append(param)
            else:
                other_params.append(param)
        
        if not train_args.with_pose:
            other_params += model['pose_encoder'].parameters()
            other_params += model['pose_decoder'].parameters()
            
        for name,param in model['depth'].named_parameters():
            if 'enc_blocks' in name:
                if train_args.encoder_freeze:
                    param.requires_grad = False
            if 'dec_blocks' in name:
                if train_args.decoder_freeze:
                    param.requires_grad = False     
        
        optimizer = torch.optim.Adam([{"params": filter(lambda p: p.requires_grad, pretrained_params), "lr":float(train_args.learning_rate)*0.1},
                                      {"params": filter(lambda p: p.requires_grad, other_params), "lr":float(train_args.learning_rate)}  ], float(train_args.learning_rate))
    # data loader
    train_ds, valv_ds, train_loader, val_loader = initialize.data_loader(train_args, train_args.batch_size, train_args.num_workers)
                                            
    # set wandb
    if train_args.log_tool == 'wandb':
        wandb.init( project = train_args.wandb_proj_name,
                    name = train_args.wandb_exp_name,
                    config = train_args,
                    dir=train_args.log_path)

    # train and val
    step = 0
    for epoch in range(train_args.num_epoch):

        # set train
        utils.model_mode(model,TRAIN)  
        
        # train loop
        tqdm_train = tqdm(train_loader, desc=f'Train Epoch: {epoch+1}/{train_args.num_epoch}')
        for i, inputs in enumerate(tqdm_train): 
                   
            # move tensors to cuda
            for key, val in inputs.items():
                if type(val) == torch.Tensor:   # not all inputs are tensors
                    inputs[key] = val.to(device)
           
            # train forward pass
            total_loss, losses, model_outs = loss.compute_loss(inputs, model, train_args, TRAIN, epoch=epoch)

            # terminal log
            tqdm_train.set_postfix({'bs':train_args.batch_size, 'train_loss':f'{total_loss:.4f}'})
            # backward pass 
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # wandb logging 
            if train_args.log_tool == 'wandb':
                wandb_dict = {"epoch":(epoch+1)}
                wandb_dict.update(losses)
                wandb.log(wandb_dict)

        # save model & optimzier (.pth)
        save_epoch_freq = int(train_args.epoch_save_freq)
        if (epoch+1) % save_epoch_freq == 0:
            print('saved model')
            utils.save_component(train_args.log_path, train_args.wandb_exp_name, epoch, model, optimizer)
        
        # load_weight_depth = torch.load('/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu0_kitti_bs16_sf_selfsup_try1_eigenzhou/weights_10/depth.pth')
        # load_weight_pose_enc = torch.load('/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu0_kitti_bs16_sf_selfsup_try1_eigenzhou/weights_10/pose_encoder.pth')
        # load_weight_pose_dec = torch.load('/media/data1/jinlovespho/log/mfdepth/pho_server5_gpu0_kitti_bs16_sf_selfsup_try1_eigenzhou/weights_10/pose_decoder.pth')

        # is_load_depth = model['depth'].module.load_state_dict(load_weight_depth)
        # is_load_pose_enc = model['pose_encoder'].module.load_state_dict(load_weight_pose_enc)
        # is_load_pose_dec = model['pose_decoder'].module.load_state_dict(load_weight_pose_dec)
        # print(is_load_depth)
        # print(is_load_pose_enc)
        # print(is_load_pose_dec)
        
        # validation
        with torch.no_grad():
            utils.model_mode(model,EVAL)
            eval_loss = 0
            eval_error = []
            pred_depths = []
            gt_depths = []
            
            inputs_color=[]

            # val loop
            tqdm_val = tqdm(val_loader, desc=f'Validation Epoch: {epoch+1}/{train_args.num_epoch}')
            for i, inputs in enumerate(tqdm_val):
                
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
                    pred_depths.extend(pred_depth_orig.squeeze(1))
                    inputs_color.extend(inputs['color',0,0].squeeze(1))
                
                else:
                    gt_depth = inputs['depth_gt']
                    gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
                    pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu().numpy())
                    
            
            if train_args.dataset == 'cityscapes':
                MIN_DEPTH = 1e-3
                MAX_DEPTH = 80
                gt_path = train_args.cs_gt_path
                num_gt_samples = len(val_loader.dataset)
                
                errors = []
                ratios = []
                vis_gt_depths=[]
                vis_pred_depths=[]
                vis_inputs=[]
                for i in range(num_gt_samples):
                    gt_depth = np.load(os.path.join(gt_path, str(i).zfill(3) + '_depth.npy'))
                    gt_height, gt_width = gt_depth.shape[:2]
                    # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
                    gt_height = int(round(gt_height * 0.75))
                    gt_depth = torch.from_numpy(gt_depth[:gt_height]).cuda()    # 768, 2048
                    pred_depth = pred_depths[i] # 768, 2048
                    
                    vis_input = inputs_color[i].unsqueeze(dim=0)
                    vis_input = F.interpolate(vis_input, (gt_height, gt_width), mode='bilinear', align_corners=True)      
                    vis_input = vis_input.squeeze(dim=0)
                    
                    # when evaluating cityscapes, we centre crop to the middle 50% of the image.
                    # Bottom 25% has already been removed - so crop the sides and the top here
                    gt_depth = gt_depth[256:, 192:1856]
                    pred_depth = pred_depth[256:, 192:1856]
                    vis_input = vis_input[:, 256:, 192:1856]
                    
                    vis_gt_depths.append(gt_depth)
                    vis_pred_depths.append(pred_depth)
                    vis_inputs.append(vis_input)

                    mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)

                    pred_depth = pred_depth[mask]
                    gt_depth = gt_depth[mask]

                    ratio = torch.median(gt_depth) / torch.median(pred_depth)
                    ratios.append(ratio)
                    pred_depth *= ratio  
                    pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
                    errors.append(compute_depth_errors(gt_depth, pred_depth)) 
                    
                ratios = torch.tensor(ratios)
                med = torch.median(ratios)
                std = torch.std(ratios / med)
                print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

                mean_errors = torch.tensor(errors).mean(0)

                print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
                print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))  
                
                depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
                error_dict = {}
                for error_name, error_value in zip(depth_metric_names, mean_errors):
                    error_dict[error_name] = error_value.item()
                error_dict["val_loss"] = eval_loss / len(val_loader)    
                
            else:                
                eval_error = eval_metric(pred_depths, gt_depths, train_args)  
                error_dict = get_eval_dict(eval_error)
                error_dict["val_loss"] = eval_loss / len(val_loader)                

            if train_args.log_tool == 'wandb':
                error_dict["epoch"] = (epoch+1)
                wandb.log(error_dict)
                if train_args.dataset == 'cityscapes':
                    visualize_cs(vis_inputs, vis_gt_depths, vis_pred_depths, model_outs, train_args)
                else:
                    visualize(inputs, pred_depth_orig, model_outs, train_args)
                
    print('End of Epoch')