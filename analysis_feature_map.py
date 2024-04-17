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
import torchvision
import numpy as np
import imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


TRAIN = 0
EVAL  = 1

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./conf/base_train_save_depth.yaml")

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
        
        with torch.no_grad():
            utils.model_mode(model,EVAL)
            eval_loss = 0
            eval_error = []
            pred_depths = []
            gt_depths = []
            # JINLOVESPHO
            pred_colors = []
            gt_colors= []
            
            train_loader, val_loader = initialize.data_loader(train_cfg.data, train_cfg.batch_size, train_cfg.num_workers)
            

            print(f'Validation progress')
            for i, inputs in enumerate(tqdm(val_loader)):
                
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
                        total_loss, _, pred_depth, full_features, pred_depth_mask = loss.compute_loss_multiframe(inputs, model, train_cfg, EVAL)   
                        
                    # for i in range(4):
                    #     feature = full_features[i].squeeze().permute(1,0)
                        
                    #     for x in range(12):
                    #         for y in range(40):
                    #             feature = F.normalize(feature, dim=1)
                    #             similarity_matrix = torch.mm(feature, feature.transpose(0, 1))
                                
                    #             similarity_matrix = similarity_matrix.reshape(12,40,480)
                    #             similarity_matrix_query = similarity_matrix[x,y]
                                
                    #             similarity_matrix_query = similarity_matrix_query.reshape(12,40)
                                
                    #             plt.figure(figsize=(10, 8))
                    #             sns.heatmap(similarity_matrix_query.cpu().detach().numpy(), cmap='viridis')
                    #             # save plt as figure
                    #             os.makedirs(f'./sim_map/similarity_matrix{i}', exist_ok=True)
                    #             plt.savefig(f'./sim_map/similarity_matrix{i}/similarity_matrix_{x}_{y}.png')
                    #             plt.close()
                                
                    
                    
                    
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

                # JINLOVESPHO
                # pred_colors.extend(pred_color.cpu().numpy())    # 굳이 color에 대해서는 eval metric 돌릴필요 없는듯.
                # gt_colors.extend(gt_color.cpu().numpy())

            # breakpoint()
            ## save pred_depths
            pred_depths = np.array(pred_depths)
            
            eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
            error_dict = get_eval_dict(eval_error)
            error_dict["val_loss"] = eval_loss / len(val_loader)                

            progress.write(f'########################### validation ###########################\n') 
            progress.write(f'{error_dict}\n') 
            progress.write(f'####################################################################################\n') 
        
        
        

    if not(train_cfg.wandb):
        progress.close()