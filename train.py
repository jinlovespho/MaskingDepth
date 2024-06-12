import os

import torch
from tqdm import tqdm
import wandb

import initialize
import utils
import loss
from eval import visualize, eval_metric, get_eval_dict

from torchvision.utils import save_image

from args_train import get_train_args
from networks.monodepth2_networks import compute_depth_losses

TRAIN = 0
EVAL  = 1

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
    encode_index = len(list(model['depth'].module.encoder.parameters()))
    optimizer = torch.optim.Adam([  {"params": params_to_train[:encode_index], "lr": 1e-5}, 
                                    {"params": params_to_train[encode_index:]},  ],              
                                 
                                 float(train_args.learning_rate))
     
    '''
    params_to_train well loaded check
    
    tot_p = sum(i.numel() for i in params_to_train) / 1e6
    
    p1=sum(i.numel() for i in model['depth'].module.parameters()) / 1e6
    p2=sum(i.numel() for i in model['pose_encoder'].module.parameters()) / 1e6
    p3=sum(i.numel() for i in model['pose_decoder'].module.parameters()) / 1e6
    
    check if tot_p = p1+p2+p3 
    '''
  
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
        if epoch+1 % save_epoch_freq == save_epoch_freq:
            utils.save_component(train_args.log_path, train_args.wandb_exp_name, epoch, model, optimizer)

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