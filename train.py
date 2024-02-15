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
        
        ############ mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        # configuration & device setting
        conf =  yaml.load(f, Loader=yaml.FullLoader)
        train_cfg = DotMap(conf['Train'])
        device = torch.device("cuda" if train_cfg.use_cuda else "cpu")
        
        # seed 
        initialize.seed_everything(train_cfg.seed)
        
        # breakpoint()
        
        #model_load
        model, parameters_to_train = initialize.baseline_model_load(train_cfg.model, device)
        model_sub, parameters_to_train_sub = initialize.additional_model_load(train_cfg.add_model, device)
        model.update(model_sub)     # 파이썬 내장 dictionary function
        parameters_to_train += parameters_to_train_sub

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
        train_loader, val_loader = initialize.data_loader(train_cfg.data, train_cfg.batch_size, train_cfg.num_workers)
                                        
        # set wandb
        if train_cfg.wandb:
            wandb.init(project = train_cfg.wandb_proj_name,
                        name = train_cfg.model_name,
                        config = conf)
        # save configuration (this part activated when do not use wandb)
        else: 
            save_config_folder = os.path.join(train_cfg.log_path, train_cfg.model_name)
            if not os.path.exists(save_config_folder):
                os.makedirs(save_config_folder)
            with open(save_config_folder + '/config.yaml', 'w') as f:
                yaml.dump(conf, f)
            progress = open(save_config_folder + '/progress.txt', 'w')
                
        
        step = 0
        print('Start Training')
        for epoch in range(train_cfg.start_epoch, train_cfg.end_epoch):
            utils.model_mode(model,TRAIN)

            # train
            print(f'Training progress(ep:{epoch+1})')
            for i, inputs in enumerate(tqdm(train_loader)): 
                
                '''
                batch_size=8일 때 
                기존에는 inputs.keys() 하면 바로 dict_keys( ['color'], ['K'] . . ) 이렇게 inputs 자체가 하나였다.
                inputs['color'].shape 을 하면 (8,3,192,640) 이었다.
                
                multiframe을 가져오록 설정하였기에, num_prev_frame=2, 즉 t-2, t-1, t 번째 frame 총 3개의 frame 을 들고와 
                이제 inputs = [ t-2_frame, t-1_frame, t_frame ] 순서 이다
                따라서 inputs[0].keys() 를 해야 dict_keys( ['color'], ['K'] . . ) 가 나오는 것. 
                
                결론은 이렇게 들어간다.
                t-2_frame => inputs[0]['curr_frame_idx'] = tensor([  31, 2803,  132,    8,  126,  336, 3495,  690])     # 여기에 t-2번째 frame에 해당하는 서로 다른 8개의 img가 sample된 것 
                t-1_frame => inputs[1]['curr_frame_idx'] = tensor([  32, 2804,  133,    9,  127,  337, 3496,  691])
                t_frame   => inputs[2]['curr_frame_idx'] = tensor([  33, 2805,  134,   10,  128,  338, 3497,  692])

                t-2_frame => inputs[0]['color'].shape = torch.Size([8, 3, 192, 640])    # 여기에는 t-2번째 frame에 해당하는 서로 다른 8개의 img 가 들어간 것 
                t-1_frame => inputs[1]['color'].shape = torch.Size([8, 3, 192, 640])
                t_frame   => inputs[2]['color'].shape = torch.Size([8, 3, 192, 640])
                
                
                이제 previous frame 도 같이 뱉도록 dataloader을 짰으니,
                이거를 어떻게 사용할지는 앞으로 어떤 실험을 돌려야할지 성훈님한테 물어보고 아래에 짜야할 듯 ?
                
                '''
                # breakpoint()
                
                # save_image(inputs['color'][0],   f'./{epoch}_{i}_{0}.png')
                # save_image(inputs['color'][1], f'./{epoch}_{i}_{1}.png')
                # save_image(inputs['color'][2], f'./{epoch}_{i}_{2}.png')
                # save_image(inputs['color'][3], f'./{epoch}_{i}_{3}.png')
                
                
                for key, ipt in inputs.items():
                    if type(ipt) == torch.Tensor:
                        inputs[key] = ipt.to(device)

                with torch.cuda.amp.autocast(enabled=True):
                    total_loss, losses = loss.compute_loss(inputs, model, train_cfg)
                
                # backward & optimizer
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if train_cfg.wandb:
                    wandb_dict = {"epoch":(epoch+1)}
                    wandb_dict.update(losses)
                    wandb.log(wandb_dict)

                else:
                    progress.write(f'(epoch:{epoch+1} / (iter:{i+1})) >> loss:{losses}\n') 
            
            # save model & optimzier (.pth)
            if epoch % 5 == 4:
                utils.save_component(train_cfg.log_path, train_cfg.model_name, epoch, model, optimizer)

            #validation
            with torch.no_grad():
                utils.model_mode(model,EVAL)
                eval_loss = 0
                eval_error = []
                pred_depths = []
                gt_depths = []

                print(f'Validation progress(ep:{epoch+1})')
                for i, inputs in enumerate(tqdm(val_loader)):
                    for key, ipt in inputs.items():
                        inputs[key] = ipt.to(device)
                    total_loss = 0
                    losses = {}

                    with torch.cuda.amp.autocast(enabled=True):
                        total_loss, _, pred_depth, pred_uncert, pred_depth_mask = loss.compute_loss(inputs, model, train_cfg, EVAL)

                    gt_depth = inputs['depth_gt']
                  
                    eval_loss += total_loss
                    pred_depths.extend(pred_depth.squeeze(1).cpu().numpy())
                    gt_depths.extend(list(gt_depth.squeeze(1).cpu().numpy()))

                eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
                error_dict = get_eval_dict(eval_error)
                error_dict["val_loss"] = eval_loss / len(val_loader)                

                if train_cfg.wandb:
                    error_dict["epoch"] = (epoch+1)
                    wandb.log(error_dict)
                    visualize(inputs, pred_depth, pred_depth_mask, pred_uncert, wandb)                    
                else:
                    progress.write(f'########################### (epoch:{epoch+1}) validation ###########################\n') 
                    progress.write(f'{error_dict}\n') 
                    progress.write(f'####################################################################################\n') 

        if not(train_cfg.wandb):
            progress.close()