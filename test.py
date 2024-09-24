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
import seaborn as sns
import cv2
import matplotlib.pyplot as plt
from flow_util import flow_to_image
from imagecorruptions import corrupt
from skimage.filters import gaussian
import PIL
from numba import njit, prange

TRAIN = 0
EVAL  = 1

@njit()
def _shuffle_pixels_njit_glass_blur(d0,d1,x,c):

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(d0 - c[1], c[1], -1):
            for w in range(d1 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]
    return x

def glass_blur(x, corruption_name='glass_blur', severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][
        severity - 1]

    x = np.uint8(
        gaussian(np.array(x) / 255., sigma=c[0]) * 255)

    x = _shuffle_pixels_njit_glass_blur(np.array(x).shape[0],np.array(x).shape[1],x,c)

    return np.clip(gaussian(x / 255., sigma=c[0]), 0,
                   1) * 255

def color_quant(x, corruption_name='color_quant', severity=1):
    x = Image.fromarray(x)

    bits = 5 - severity + 1
    x = PIL.ImageOps.posterize(x, bits)
    return np.array(x)

def iso_noise(x, corruption_name='iso_noise', severity=1):
    c_poisson = 25
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity-1]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
    return x



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
    parser.add_argument('--attn_conv4d', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0.0)
    parser.add_argument("--moving_masking", type=str, default="None")
    parser.add_argument('--attn_agg_tf', action='store_true')
    parser.add_argument('--img_recon_weight', type=float, default=0.0)
    parser.add_argument("--augmentation", type=str, default=None)
    parser.add_argument("--severity", type=int, default=1)
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

            # img = (inputs['color', 0, 0].squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            # if train_args.augmentation == 'glass_blur':
            #     corrupted = glass_blur(img, severity=train_args.severity)/255.
            # elif train_args.augmentation == 'color_quant':
            #     corrupted = color_quant(img, severity=train_args.severity)/255.
            # elif train_args.augmentation == 'iso_noise':
            #     corrupted = iso_noise(img, severity=train_args.severity)/255.
            # else:
            #     corrupted = corrupt(img, corruption_name=train_args.augmentation, severity=train_args.severity)/255.
            # inputs['color', 0, 0] = torch.from_numpy(corrupted).unsqueeze(0).permute(0,3,1,2).float().to(device)

            img = (inputs['color', -1, 0].squeeze().permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

            if train_args.augmentation == 'glass_blur':
                corrupted = glass_blur(img, severity=train_args.severity)/255.
            elif train_args.augmentation == 'color_quant':
                corrupted = color_quant(img, severity=train_args.severity)/255.
            elif train_args.augmentation == 'iso_noise':
                corrupted = iso_noise(img, severity=train_args.severity)/255.
            else:
                corrupted = corrupt(img, corruption_name=train_args.augmentation, severity=train_args.severity)/255.
            inputs['color', -1, 0] = torch.from_numpy(corrupted).unsqueeze(0).permute(0,3,1,2).float().to(device)

            # val forward pass
            total_loss, losses, pred_depth_orig, model_outs = loss.compute_loss(inputs, model, train_args, EVAL)

            eval_loss += total_loss
            
            gt_depth = inputs['depth_gt']
            pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu().numpy())
            gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
        
        eval_error = eval_metric(pred_depths, gt_depths, train_args)  
        error_dict = get_eval_dict(eval_error)
        error_dict["val_loss"] = eval_loss / len(val_loader)   

        ## save in txt file
        with open(f'./val_error_t-1.txt', 'a') as f:
            f.write(f"{train_args.augmentation} {train_args.severity}: {str(error_dict)} + \n")
                 

        if train_args.log_tool == 'wandb':
            error_dict["epoch"] = (1)
            wandb.log(error_dict)
            visualize(inputs, pred_depth_orig, model_outs, train_args)
                
    print('End of Epoch')