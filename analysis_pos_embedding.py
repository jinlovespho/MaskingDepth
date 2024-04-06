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
        encode_key = model['depth'].state_dict().keys()
        encode_index = [True if 'encoder' in key else False for key in encode_key]
        other_index = [False if 'encoder' in key else True for key in encode_key]
        # encode_index = len(list(model['depth'].module.encoder.parameters()))
        optimizer = optim.Adam([{"params": parameters_to_train[encode_index], "lr": 1e-5}, 
                                {"params": parameters_to_train[other_index]}], float(train_cfg.lr))
        import pdb;pdb.set_trace()
        

        ## make correlation map between pose_embedding1 and pose_embedding1(x,y)
        
        ## x 0~12. y 0~40 all pairs
        
        # for x in range(12):
        #     for y in range(40):
        #         pose_embedding1 = F.normalize(pose_embedding1, dim=-1)
        #         similarity_matrix = torch.mm(pose_embedding1, pose_embedding1.transpose(0, 1))
                
                
        #         similarity_matrix = similarity_matrix.reshape(12,40,480)
        #         similarity_matrix_query = similarity_matrix[x,y]
                
        #         similarity_matrix_query = similarity_matrix_query.reshape(12,40)
                
        #         plt.figure(figsize=(10, 8))
        #         sns.heatmap(similarity_matrix_query.cpu().detach().numpy(), cmap='viridis')
        #         # save plt as figure
        #         plt.savefig(f'./similarity_matrix/similarity_matrix_1_{x}_{y}.png')
        for x in range(12):
            for y in range(40):
                pose_embedding2 = F.normalize(pose_embedding2, dim=-1)
                similarity_matrix = torch.mm(pose_embedding2, pose_embedding2.transpose(0, 1))
                
                
                similarity_matrix = similarity_matrix.reshape(12,40,480)
                similarity_matrix_query = similarity_matrix[x,y]
                
                similarity_matrix_query = similarity_matrix_query.reshape(12,40)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(similarity_matrix_query.cpu().detach().numpy(), cmap='viridis')
                # save plt as figure
                plt.savefig(f'./similarity_matrix2/similarity_matrix_2_{x}_{y}.png')
        import pdb;pdb.set_trace()
        
        
        

    if not(train_cfg.wandb):
        progress.close()