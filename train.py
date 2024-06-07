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


TRAIN = 0
EVAL  = 1

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./conf/base_train.yaml")

args = parser.parse_args()


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
        for epoch in range(train_cfg.start_epoch, train_cfg.end_epoch):
            utils.model_mode(model,TRAIN)  
            
            # train
            tqdm_train = tqdm(train_loader, desc=f'Train Epoch: {epoch+1}/{train_cfg.end_epoch}')
            for i, inputs in enumerate(tqdm_train): 
                
                if train_cfg.data.dataset=='kitti_depth_multiframe':
                    for input in inputs:
                        for key, ipt in input.items():
                            if type(ipt) == torch.Tensor:
                                input[key] = ipt.to(device)     # Place current and previous frames on cuda  
                    
                    total_loss, losses = loss.compute_loss_multiframe(inputs, model, train_cfg, TRAIN)     
                                
                else:
                    for key, ipt in inputs.items():
                        if type(ipt) == torch.Tensor:
                            inputs[key] = ipt.to(device)
                    # with torch.cuda.amp.autocast(enabled=True):
                    total_loss, losses = loss.compute_loss(inputs, model, train_cfg, TRAIN)
                
                tqdm_train.set_postfix({'train_loss':f'{total_loss:.4f}'})
                # backward & optimizer
                optimizer.zero_grad()
                # scaler.scale(total_loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                total_loss.backward()
                optimizer.step()
                
                # wandb logging 
                if train_cfg.wandb:
                    wandb_dict = {"epoch":(epoch+1)}
                    wandb_dict.update(losses)
                    wandb.log(wandb_dict)

                else:
                    progress.write(f'(epoch:{epoch+1} / (iter:{i+1})) >> loss:{losses}\n') 
            
            # save model & optimzier (.pth)
            save_epoch_freq = int(train_cfg.save_epoch_freq)
            if epoch+1 % save_epoch_freq == save_epoch_freq:
                utils.save_component(train_cfg.log_path, train_cfg.model_name, epoch, model, optimizer)

            #validation
            with torch.no_grad():
                utils.model_mode(model,EVAL)
                eval_loss = 0
                eval_error = []
                pred_depths = []
                gt_depths = []

                tqdm_val = tqdm(val_loader, desc=f'Validation Epoch: {epoch+1}/{train_cfg.end_epoch}')
                for i, inputs in enumerate(tqdm_val):
                    
                    total_loss = 0
                    losses = {}
                    
                    # multiframe validation
                    if train_cfg.data.dataset=='kitti_depth_multiframe':
                        for input in inputs:
                            for key, ipt in input.items():
                                if type(ipt) == torch.Tensor:
                                    input[key] = ipt.to(device)     # Place current and previous frames on cuda

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
                
                    eval_loss += total_loss

                    pred_depths.extend(pred_depth.squeeze(1).cpu().numpy())
                    gt_depths.extend(gt_depth.squeeze(1).cpu().numpy())

                eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
                error_dict = get_eval_dict(eval_error)
                error_dict["val_loss"] = eval_loss / len(val_loader)                

                if train_cfg.wandb:
                    error_dict["epoch"] = (epoch+1)
                    wandb.log(error_dict)
                    if train_cfg.data.dataset=='kitti_depth_multiframe':
                        visualize(inputs[0], pred_depth, pred_depth_mask, pred_uncert, wandb) 
                    else:
                        visualize(inputs, pred_depth, pred_depth_mask, pred_uncert, wandb)  
                                   
                else:
                    progress.write(f'########################### (epoch:{epoch+1}) validation ###########################\n') 
                    progress.write(f'{error_dict}\n') 
                    progress.write(f'####################################################################################\n') 

        if not(train_cfg.wandb):
            progress.close()