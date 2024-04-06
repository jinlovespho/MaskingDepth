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
                                        
                
        fpath = os.path.join(os.path.dirname(__file__), "splits", train_cfg.data.splits, "{}_files.txt")

        train_filenames = utils.readlines(fpath.format("val"))

        #validation
        with torch.no_grad():
            utils.model_mode(model,EVAL)
            eval_loss = 0
            eval_error = []
            pred_depths = []
            gt_depths = []
            # JINLOVESPHO
            pred_colors = []
            gt_colors= []

            print(f'Validation progress')
            for i, inputs in enumerate(tqdm(train_loader)):
                pass
                
                
                
    if not(train_cfg.wandb):
        progress.close()