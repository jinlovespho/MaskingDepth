import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import numpy as np
from PIL import Image 
import torchvision

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


TRAIN = 0
EVAL  = 1

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    x_normal = np.linspace(-1,1,40)
    x_normal = torch.tensor(x_normal, dtype=torch.float, requires_grad=False)
    y_normal = np.linspace(-1,1,12)
    y_normal = torch.tensor(y_normal, dtype=torch.float, requires_grad=False)
    
    b,_,h,w = corr.size()
    
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:,0,:,:] = (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0 # unormalise
    mapping[:,1,:,:] = (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0 # unormalise

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if mapping.is_cuda:
        grid = grid.cuda()
    flow = mapping - grid
    return flow

def flow2bearing(flow, fc, cc, normalize=True):
    assert len(flow.shape) == 4
    height, width = flow.shape[2:4]
    xx, yy = np.meshgrid(range(width), range(height))
    pixel = torch.zeros_like(flow)
    match = (flow[:, 0, ...] + torch.from_numpy(xx), flow[:, 1] + torch.from_numpy(yy))
    pixel[:, 0] = (match[0] - cc[0]) / fc[0]
    pixel[:, 1] = (match[1] - cc[1]) / fc[1]
    pixel = torch.cat((pixel, torch.ones_like(pixel[:, 0:1])), dim=1)

    if normalize:
        pixel = F.normalize(pixel)
    return pixel

def rot_bearing_mul(rot, bearing):
    # rot: B x 3 x 3, bearing: B x 3 x H x W
    product = torch.bmm(rot, bearing.view(bearing.shape[0], 3, -1))
    return product.view(bearing.shape)

def ls_2view(r, s):
    hessian = (s * s).sum(dim=1, keepdims=True)
    z = -(s * r).sum(dim=1, keepdims=True) / (hessian + 1e-30)
    e = (r * r).sum(dim=1, keepdims=True) - hessian * (z ** 2)

    invalid_mask = (z <= 0.1)
    invalid_mask |= (z >= 10)
    # invalid_mask |= (e > 0.015 ** 2)
    z[invalid_mask] = 0
    e[invalid_mask] = 0
    hessian[invalid_mask] = 0
    return z, e, hessian

def rot_bearing_mul(rot, bearing):
    # rot: B x 3 x 3, bearing: B x 3 x H x W
    product = torch.bmm(rot, bearing.view(bearing.shape[0], 3, -1))
    return product.view(bearing.shape)

def triangulation( bearings_ref_in_other, t_ref_in_other, flows, residual=False):
    rs, ss = pre_triangulation(bearings_ref_in_other, t_ref_in_other, flows, concat=False)
    # get output = (z, residual, hessian)
    outputs = [ls_2view(*r_s) for r_s in zip(rs, ss)]
    # weighted sum of z with weight hessian
    hessian = sum([output[2] for output in outputs])
    pred_depths = sum([output[0] * output[2] for output in outputs]) / (hessian + 1e-12)

    if residual:
        # hessian*(z* - z)^2 + residual
        error = torch.sqrt(
            sum([output[2] * (pred_depths - output[0]) ** 2 + output[1] for output in outputs]).clamp_min(0))
        sqrt_hessian = torch.sqrt(hessian)
        return pred_depths, (error, sqrt_hessian)
    else:
        return pred_depths

def pre_triangulation(bearings_ref_in_other, t_ref_in_other, flows, concat=True):
    resize=16
    fc = np.array([371.2000,368.6400]) / resize
    cc = np.array([320., 96.]) / resize
    bearings_other = [flow2bearing(flow, fc, cc, normalize=True) for flow in flows]
    ss = [torch.cross(bearings_other[k], bearings_ref_in_other[k], dim=1) for k in
                range(len(bearings_other))]
    rs = [torch.cross(bearings_other[k], t_ref_in_other[:, k, :, None, None].expand_as(bearings_other[k]), dim=1)
            for k in range(len(bearings_other))]

    if concat:
        s = torch.cat(ss, dim=1)
        r = torch.cat(rs, dim=1)
        return r, s
    else:
        return rs, ss

def generate_image_homogeneous_coordinates(fc, cc, image_width, image_height):
    homogeneous = np.zeros((image_height, image_width, 3))
    homogeneous[:, :, 2] = 1

    xx, yy = np.meshgrid([i for i in range(0, image_width)], [i for i in range(0, image_height)])
    homogeneous[:, :, 0] = (xx - cc[0]) / fc[0]
    homogeneous[:, :, 1] = (yy - cc[1]) / fc[1]

    return torch.from_numpy(homogeneous.astype(np.float32))



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
    parser.add_argument('--moving_masking', type=str, default='None')
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

    # set hooks to visualize cross attention maps in decoder
    ca1_names=[]
    ca1_modules=[]
    ca1_maps=[]
    for name, module in model['depth'].named_modules():
        if 'dec_blocks' in name and 'cross_attn.attn_drop' in name:
            ca1_names.append(name)
            ca1_modules.append(module)
            module.register_forward_hook(lambda m,i,o: ca1_maps.append(o.detach().cpu()) )

    sa1_names=[]
    sa1_modules=[]
    sa1_maps=[]
    for name, module in model['depth'].named_modules():
        if 'dec_blocks' in name and 'cross_attn' not in name and 'attn.attn_drop' in name:
            sa1_names.append(name)
            sa1_modules.append(module)
            module.register_forward_hook(lambda m,i,o: sa1_maps.append(o.detach().cpu()) )
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
        
            # val forward pass
            total_loss, losses, pred_depth_orig, model_outs = loss.compute_loss(inputs, model, train_args, EVAL)
            
            inputs['color',-1,0] = inputs['color',0,0]
            _,_,pred_depth_tt,_ =  loss.compute_loss(inputs, model, train_args, EVAL)
            
            # attn map visualize
            img_curr = inputs['color',0,0]      # b 3 192 640
            img_prev = inputs['color',-1,0]
            
            sa1_map = torch.stack(sa1_maps, dim=1)  # b num_layer num_head N1 N2
            ca1_map = torch.stack(ca1_maps, dim=1)  # b num_layer num_head N1 N2
            
            sa1_maps.clear()
            ca1_maps.clear()
            
            sa1_map = sa1_map.mean(dim=2)   # b num_layer N1 N2
            ca1_map = ca1_map.mean(dim=2)   # b num_layer N1 N2
            
            # log_name=f'.vis_attn/{train_args.model_info}/iter{k}'
            log_name = "moving_object"
            if not os.path.exists(log_name):
                os.makedirs(log_name)
                
            vis_num_points=0
            rnd = torch.rand(480).argsort()
            
            pred_depth_orig = pred_depth_tt
            
            # abs_error = torch.abs(pred_depth_orig - pred_depth_tt) / pred_depth_tt
            # mask = abs_error<0.1
            # ## visualize mask
            # torchvision.utils.save_image(mask.float(), f'{log_name}/{k}_mask.jpg', normalize=True)
            
            # i_img_curr=img_curr[0].detach().cpu()
            # i_img_curr_np = np.float32(i_img_curr.permute(1,2,0)) * 255
            # i_img_curr_np = i_img_curr_np[:,:,[2,1,0]]
            # cv2.imwrite(f'{log_name}/{k}_img_curr.jpg', i_img_curr_np)
            
            # vis_pred_depth = pred_depth_orig[0].squeeze().cpu().numpy()     # 1 375 1242
            # vmax = np.percentile(vis_pred_depth, 95)
            # normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
            # colormapped_pred_depth = cv2.resize(colormapped_pred_depth, (640,192))
            # vis_pred_depth = Image.fromarray(colormapped_pred_depth)
            # vis_pred_depth.save(f'{log_name}/{k}_pred_depth_t1.jpg')
            
            # vis_pred_depth = pred_depth_tt[0].squeeze().cpu().numpy()     # 1 375 1242
            # vmax = np.percentile(vis_pred_depth, 95)
            # normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
            # colormapped_pred_depth = cv2.resize(colormapped_pred_depth, (640,192))
            # vis_pred_depth = Image.fromarray(colormapped_pred_depth)
            # vis_pred_depth.save(f'{log_name}/{k}_pred_depth_t.jpg')
            
            ## visualize mask
            
            # matching = ca1_map[0].mean(dim=0)
            # matching[:,441] = 0
            # output_shape=(192,640)
            
            
            # grid_x, grid_y = soft_argmax(matching.unsqueeze(dim=0).transpose(-2,-1).reshape(1,-1,12,40),beta=2e-4)
            # flow = torch.cat((grid_x, grid_y), dim=1)
            
            # flow = unnormalise_and_convert_mapping_to_flow(flow)
            # height, width = flow.shape[-2:]
            # # flow = F.interpolate(flow, size=output_shape, mode='bilinear', align_corners=False)
            # # flow[:, 0] *= float(output_shape[1]) / float(width)
            # # flow[:, 1] *= float(output_shape[0]) / float(height)
            
            # # model_outs['pose_tmp'] = torch.linalg.inv(model_outs['pose_tmp'])
            # rots = model_outs['pose_tmp'][:,:3,:3]
            # ts = model_outs['pose_tmp'][:,:3,3]
            
            # resize=16
            # fc = np.array([371.2000,368.6400]) / resize
            # cc = np.array([320., 96.]) / resize
            # image_size = (640 // resize, 192 // resize)
            # homogeneous_coords = generate_image_homogeneous_coordinates(fc, cc, *image_size).permute(2, 0, 1)
            # bearings_ref_in_other = rot_bearing_mul(rots.cpu(), homogeneous_coords.cpu().unsqueeze(dim=0))
            # pred_depth, other = triangulation(bearings_ref_in_other.cpu().unsqueeze(dim=1), ts.cpu().unsqueeze(dim=1), flow.cpu().unsqueeze(dim=1), residual=True)

            # vis_pred_depth = pred_depth.squeeze().cpu().numpy()     # 1 375 1242
            # vmax = np.percentile(vis_pred_depth, 95)
            # normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
            # colormapped_pred_depth = cv2.resize(colormapped_pred_depth, (640,192))
            # vis_pred_depth = Image.fromarray(colormapped_pred_depth)
            
            # vis_pred_depth.save(f'{log_name}/tri_depth_{k}.jpg')
            

            # flow_img = flow_to_image(flow.squeeze().permute(1,2,0).cpu().numpy())
            # flow_img = Image.fromarray(flow_img)
            # flow_img.save(f'{log_name}/flow.jpg')
            
            # sa1_map_mean = sa1_map[0].mean(dim=0)
            # ca1_map_mean = ca1_map[0].mean(dim=0)
            # ca1_map_mean[:,441]=0
            
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(sa1_map_mean, cmap='viridis')
            # plt.savefig(f'{log_name}/{k}_sa1_map.jpg')
            # plt.close()
            
            # plt.figure(figsize=(10, 8))
            # sns.heatmap(ca1_map_mean, cmap='viridis')
            # plt.savefig(f'{log_name}/{k}_ca1_map.jpg')
            # plt.close()
            
            # i_img_curr=img_curr[0].detach().cpu()  # 3 192 640
            # i_img_prev=img_prev[0].detach().cpu()
            
            # ## visualize i_img_curr
            # i_img_curr_np = np.float32(i_img_curr.permute(1,2,0)) * 255
            # i_img_curr_np = i_img_curr_np[:,:,[2,1,0]]
            # cv2.imwrite(f'{log_name}/{k}_img_curr.jpg', i_img_curr_np)
            
            # ## visualize i_img_prev
            # i_img_prev_np = np.float32(i_img_prev.permute(1,2,0)) * 255
            # i_img_prev_np = i_img_prev_np[:,:,[2,1,0]]
            # cv2.imwrite(f'{log_name}/{k}_img_prev.jpg', i_img_prev_np)
            
            # vis_pred_depth = pred_depth_orig[0].squeeze().cpu().numpy()     # 1 375 1242
            # vmax = np.percentile(vis_pred_depth, 95)
            # normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
            # colormapped_pred_depth = cv2.resize(colormapped_pred_depth, (640,192))
            # vis_pred_depth = Image.fromarray(colormapped_pred_depth)
            # vis_pred_depth.save(f'{log_name}/{k}_pred_depth.jpg')
            
            
            # for tkn_vis_idx in rnd[:vis_num_points]:
            #     tkn_vis_idx=tkn_vis_idx.item()
            #     vis_idx_h = int(tkn_vis_idx//40 * 16)
            #     vis_idx_w = int(tkn_vis_idx%40 * 16)
            #     pt1 = (vis_idx_w, vis_idx_h)
            #     pt2 = (vis_idx_w+16, vis_idx_h+16)
        
            #     for i in range(train_args.batch_size):  # batch별
                    
            #         i_img_curr=img_curr[i].detach().cpu()  # 3 192 640
            #         i_img_prev=img_prev[i].detach().cpu()
            #         i_sa1_map=sa1_map[i].detach().cpu()    # 12 480 480
            #         i_ca1_map=ca1_map[i].detach().cpu()    # num_layer N1 N2
                    
            #         vis_pred_depth = pred_depth_orig[i].squeeze().cpu().numpy()     # 1 375 1242
            #         vmax = np.percentile(vis_pred_depth, 95)
            #         normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
            #         mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            #         colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
            #         colormapped_pred_depth = cv2.resize(colormapped_pred_depth, (640,192))
            #         vis_pred_depth = Image.fromarray(colormapped_pred_depth)
                      
            #         # save_image(i_img_curr, f'{log_name}/{i}_img_curr.jpg')
            #         # save_image(i_img_prev, f'{log_name}/{i}_img_prev.jpg')
                    
            #         # log current image and query point
            #         i_img_curr_np = np.float32(i_img_curr.permute(1,2,0)) * 255 
            #         i_img_curr_np = i_img_curr_np[:,:,[2,1,0]]
            #         i_img_curr_query=cv2.rectangle(i_img_curr_np.astype(np.uint8).copy(), pt1, pt2, (0,0,255), -1)
            #         log_name2=f'{log_name}/tkn{tkn_vis_idx}_h{vis_idx_h}_w{vis_idx_w}'
            #         if not os.path.exists(log_name2):
            #             os.makedirs(log_name2)
            #         file_name=f'batch{i}_img_curr.jpg'
            #         cv2.imwrite(f'{log_name2}/aaa_tmp_curr{i}.jpg', i_img_curr_query) 
            #         cv2.imwrite(f'{log_name2}/{file_name}', i_img_curr_query)  
            #         vis_pred_depth.save(f'{log_name2}/{file_name}_pred_depth.jpg')
            #         i_ca1_map[:,:,441] = 0
            #         # for j in range(12):     # layer별
            #         j_sa1_map=i_sa1_map.mean(dim=0)  # 480 480
            #         j_ca1_map=i_ca1_map.mean(dim=0)  # N1 N2 
                    
            #         vis_tkn_sa1 = j_sa1_map[tkn_vis_idx]    # 480
            #         vis_tkn_ca1 = j_ca1_map[tkn_vis_idx]
                    
            #         interp_vis_tkn_sa1 = F.interpolate(vis_tkn_sa1.view(12,40).unsqueeze(dim=0).unsqueeze(dim=0), size=(192,640), mode='bilinear', align_corners=True)  # 1 1 192 640
            #         interp_vis_tkn_ca1 = F.interpolate(vis_tkn_ca1.view(12,40).unsqueeze(dim=0).unsqueeze(dim=0), size=(192,640), mode='bilinear', align_corners=True)
                    
            #         # normalize attn map weight to [0,1]
            #         interp_vis_tkn_sa1 = (interp_vis_tkn_sa1 - interp_vis_tkn_sa1.min()) / (interp_vis_tkn_sa1.max() - interp_vis_tkn_sa1.min())
            #         interp_vis_tkn_ca1 = (interp_vis_tkn_ca1 - interp_vis_tkn_ca1.min()) / (interp_vis_tkn_ca1.max() - interp_vis_tkn_ca1.min())
                    
            #         # show attention map on previous image
            #         result_sa = show_mask_on_image(i_img_prev.permute(1,2,0), interp_vis_tkn_sa1.squeeze())    # 192 640 3
            #         result_ca = show_mask_on_image(i_img_prev.permute(1,2,0), interp_vis_tkn_ca1.squeeze())    # 192 640 3
                    
            #         # cv2.imwrite(f'{log_name}/iter{k}_batch{i}_h{vis_idx_h}_w{vis_idx_w}_img_prev_SA_layer{j}.jpg', result_sa)
            #         cv2.imwrite(f'{log_name2}/batch{i}_img_prev_SA_layer.jpg', result_sa)
            #         cv2.imwrite(f'{log_name2}/batch{i}_img_prev_CA_layer.jpg', result_ca)
                        
            #             # save_image(interp_vis_tkn_sa1.squeeze(), f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_interp_sa_map_n.jpg', normalize=True)
            #             # save_image(interp_vis_tkn_ca1.squeeze(), f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_interp_ca_map_n.jpg', normalize=True)
                        
            #             # save_image(vis_tkn_sa1, f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_sa_map_n.jpg', normalize=True)
            #             # save_image(vis_tkn_ca1, f'{log_name}/{i}_layer{j}_loc{vis_idx_h}_{vis_idx_w}_ca_map_n.jpg', normalize=True)
                    
            #     #     print('visualized per layer')
            #     # print('visualized per point')
            # # print('visualized')

            eval_loss += total_loss
            
            gt_depth = inputs['depth_gt']
            pred_depths.extend(pred_depth_orig.squeeze(1).detach().cpu().numpy())
            gt_depths.extend(gt_depth.squeeze(1).detach().cpu().numpy())
        
        eval_error = eval_metric(pred_depths, gt_depths, train_args)  
        error_dict = get_eval_dict(eval_error)
        error_dict["val_loss"] = eval_loss / len(val_loader)   
                 

        if train_args.log_tool == 'wandb':
            error_dict["epoch"] = (1)
            wandb.log(error_dict)
            visualize(inputs, pred_depth_orig, model_outs, train_args)
                
    print('End of Epoch')