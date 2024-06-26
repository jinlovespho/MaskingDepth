import argparse
import os
import torch
from tqdm import tqdm
import wandb

import initialize
import utils
import loss
from eval import visualize, eval_metric, get_eval_dict

from torchvision.utils import save_image
from networks.monodepth2_networks import compute_depth_losses


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
    # Training args 
    parser.add_argument('--num_epoch',      type=int)  
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--backbone_lr',    type=float)
    parser.add_argument('--lr',             type=float) 
    parser.add_argument('--num_workers',    type=int) 
    parser.add_argument('--seed',           type=int)
    # Depth args 
    parser.add_argument('--min_depth',      type=float,     default=0.1)
    parser.add_argument('--max_depth',      type=float,     default=80.0)
    # Loss args
    parser.add_argument('--training_loss',  type=str)
    parser.add_argument('--use_future_frame',   action='store_true')
    # Model args 
    parser.add_argument('--model_info',             type=str)
    parser.add_argument('--vit_type',               type=str,   default='vit_base')
    parser.add_argument('--pretrained_weight',      type=str)
    parser.add_argument('--pretrained_weight_path', type=str)
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
    parser.add_argument('--load_weight_path', type=str)
    args = parser.parse_args()
    return args


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
    # encode_index = len(list(model['depth'].module.encoder.parameters()))
    # optimizer = torch.optim.Adam([  {"params": params_to_train[:encode_index], "lr": train_args.backbone_lr}, 
    #                                 {"params": params_to_train[encode_index:]} ],                                        
    #                                 train_args.learning_rate)
    
    optimizer = torch.optim.Adam(params_to_train, train_args.lr)
    
    # data loader
    train_ds, val_ds, train_loader, val_loader = initialize.data_loader(train_args, train_args.batch_size, train_args.num_workers)
                                            
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
            total_loss, losses, model_outs = loss.compute_loss(inputs, model, train_args, TRAIN)

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
                total_loss, losses, pred_depth_orig, model_outs = loss.compute_loss(inputs, model, train_args, EVAL)
                
                eval_loss += total_loss
                
                gt_depth = inputs['depth_gt']
                pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu().numpy())
                gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
             
            eval_error = eval_metric(pred_depths, gt_depths, train_args)  
            error_dict = get_eval_dict(eval_error)
            error_dict["val_loss"] = eval_loss / len(val_loader)                

            if train_args.log_tool == 'wandb':
                error_dict["epoch"] = (epoch+1)
                wandb.log(error_dict)
                visualize(inputs, pred_depth_orig, model_outs, train_args)
                
    print('End of Epoch')