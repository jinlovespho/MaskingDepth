import torch
import torch.nn.functional as F
import torchvision 
import utils
from utils import * 

import sys
import pdb
import random
import wandb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



TRAIN   = 0
EVAL    = 1

# total loss
def compute_loss_twice(inputs, model, train_args, mode = TRAIN,epoch=0):
    losses = {}
    total_loss = 0
    
    orig_h, orig_w = inputs['depth_gt'].shape[-2:]
    gt_depth = inputs['depth_gt']
    
    # fwd pass TWICE
    model_outs1, model_outs2 = model_forward_twice(inputs, model, train_args, mode) 
    
    # self-supervised training
    if train_args.training_loss == 'selfsupervised_img_recon': 
        
        # t, t
        recon_losses1 = []
        smooth_losses1 = []

        fa1,ft1,ba1,bt1 = pose_forward(inputs, model)
        front_pose, back_pose = None, None

        recon_loss1, mask1, _,smooth_loss1 = compute_selfsup_mono_loss1(model_outs1, inputs, train_args, fa1, ft1,ba1,bt1, front_pose=front_pose, back_pose=back_pose)
        recon_losses1.append(recon_loss1)
        smooth_losses1.append(smooth_loss1)
        
        pred_depth_orig1 = F.interpolate(model_outs1['pred_depth',0,0], (orig_h, orig_w), mode="bilinear", align_corners = True)   # (b,1,375,1242)
        
        losses['selfsup_loss1'] = torch.stack(recon_losses1).mean()
        losses['smooth_loss1'] = torch.stack(smooth_losses1).mean()
        
        
        # t, t-1
        recon_losses2 = []
        smooth_losses2 = []

        fa2,ft2,ba2,bt2 = pose_forward(inputs, model)
        front_pose, back_pose = None, None

        recon_loss2, mask2, _,smooth_loss2 = compute_selfsup_mono_loss2(model_outs2, inputs, train_args, fa2, ft2, ba2, bt2, front_pose=front_pose, back_pose=back_pose, model_outs1=model_outs1, mode=mode)
        recon_losses2.append(recon_loss2)
        smooth_losses2.append(smooth_loss2)
        
        pred_depth_orig2 = F.interpolate(model_outs2['pred_depth',0,0], (orig_h, orig_w), mode="bilinear", align_corners = True)   # (b,1,375,1242)
        
        losses['selfsup_loss2'] = torch.stack(recon_losses2).mean()
        losses['smooth_loss2'] = torch.stack(smooth_losses2).mean()

    else:
        pass
        
  
    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    # returns
    if mode == TRAIN:
        return total_loss, losses, model_outs1, model_outs2
    else:
        return total_loss, losses, pred_depth_orig1, pred_depth_orig2, model_outs1, model_outs2


def model_forward_twice(inputs, model, train_args, mode, with_pose = False):
    
    if mode == TRAIN:
        curr_frame = inputs['color_aug',0,0]
        prev_frame = inputs['color_aug',-1,0]
        outputs1 = model['depth'](curr_frame, curr_frame, mode, intrinsics=inputs['K',0])   # t, t     
        outputs2 = model['depth'](curr_frame, prev_frame, mode, intrinsics=inputs['K',0])   # t, t-1
            
    else:
        curr_frame = inputs['color',0,0]
        prev_frame = inputs['color',-1,0]
        outputs1 = model['depth'](curr_frame, curr_frame, mode, intrinsics=inputs['K',0])   # t, t              
        outputs2 = model['depth'](curr_frame, prev_frame, mode, intrinsics=inputs['K',0])   # t, t-1  
    
    return outputs1, outputs2 
        

def pose_forward(inputs, model):
    
    # ForkedPdb().set_trace()
    pose_inputs = [model["pose_encoder"](torch.cat( [inputs['color',-1,0], inputs['color',0,0] ], 1))]
    fa,ft = model['pose_decoder'](pose_inputs)
    
    pose_inputs = [model["pose_encoder"](torch.cat( [inputs['color',0,0], inputs['color',1,0] ], 1))]
    ba,bt = model['pose_decoder'](pose_inputs)

    return fa,ft,ba,bt


############################################################################## 
########################    loss function set
############################################################################## 
  
def compute_sup_loss(pred_depth, gt_depth, non_zero_mask):
    if non_zero_mask == None:
        loss = torch.abs(pred_depth - gt_depth.detach()).mean()
    else:
        loss = torch.abs(pred_depth[non_zero_mask] - gt_depth.detach()[non_zero_mask]).mean()
    return loss


def compute_selfsup_mono_loss1(model_outs, inputs, train_args, angle, trans, back_angle, back_trans, front_pose=None, back_pose=None):
# def compute_selfsup_mono_loss(label_pred_depth, label, train_args, angle, trans, back_angle, back_trans, scale_disp):
    
    loss = 0
    loss_records = 0
    smooth_losses = 0
    device = model_outs['pred_disp',0].device

    color = inputs['color',0,0]
    target = inputs['color',0,0]

    backproject_depth = utils.BackprojectDepth(train_args.batch_size, train_args.re_height, train_args.re_width)
    backproject_depth.to(device)
    project_3d = utils.Project3D(train_args.batch_size, train_args.re_height, train_args.re_width)
    project_3d.to(device)

    front_T = utils.transformation_from_parameters(angle[:, 0], trans[:, 0], invert=(-1<0))
    back_T = utils.transformation_from_parameters(back_angle[:,0],back_trans[:,0],invert=(1<0))

    for scale in range(4):
        reprojection_losses = []

        disp = model_outs['pred_disp',scale]
        ## resize disp to target_size
        disp = F.interpolate(disp, target.shape[-2:], mode="bilinear", align_corners = False)
        _, depth = utils.disp_to_depth(disp, train_args.min_depth, train_args.max_depth)
        
        model_outs['pred_depth',0, scale] = depth

        ## back to current
        cam_points = backproject_depth(depth, inputs['inv_K',0])
        pix_coords = project_3d(cam_points, inputs['K',0], front_T)

        front_repoj_image = F.grid_sample(inputs['color',-1,0],pix_coords.to(torch.float32),padding_mode="border")
        
        ## front to back
        if train_args.use_future_frame:
            cam_points = backproject_depth(depth, inputs['inv_K',0])
            pix_coords = project_3d(cam_points, inputs['K',0], back_T)

        back_repoj_image = F.grid_sample(inputs['color',1,0],pix_coords.to(torch.float32),padding_mode="border")
        
        if scale == 0:
            model_outs['reproj_img_from_prev'] = front_repoj_image
            model_outs['reproj_img_from_fut'] =  back_repoj_image
                   
        reprojection_losses.append(utils.compute_reprojection_loss(front_repoj_image, target))
        
        # JINLOVESPHO - future frame 도 사용
        if train_args.use_future_frame:
            reprojection_losses.append(utils.compute_reprojection_loss(back_repoj_image, color))

        reprojection_losses = torch.cat(reprojection_losses, 1)
        reprojection_loss = reprojection_losses

        # ## auto masking
        identity_reprojection_losses = []
        identity_reprojection_losses.append(
            utils.compute_reprojection_loss(inputs['color',-1,0], target))
        
        # JINLOVESPHO use future frame
        if train_args.use_future_frame:
            identity_reprojection_losses.append(
                utils.compute_reprojection_loss(inputs['color',1,0], target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
        identity_reprojection_loss = identity_reprojection_losses

        identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape, device=device) * 0.00001

        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

        mask = torch.argmin(combined, dim=1).unsqueeze(1).float()
        mask[mask>=1] = 1.0
        loss_record = mask * reprojection_loss / mask.sum().detach()
        loss_records += loss_record.mean().detach()

        to_optimise, idxs = torch.min(combined, dim=1)

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = utils.get_smooth_loss(norm_disp, target)

        smooth_losses += train_args.smooth_weight * smooth_loss / (2 ** scale)

    return loss/4.0, mask, loss_records, smooth_losses/4.0



def compute_selfsup_mono_loss2(model_outs, inputs, train_args, angle, trans, back_angle, back_trans, front_pose=None, back_pose=None, model_outs1=None, mode=None):
# def compute_selfsup_mono_loss(label_pred_depth, label, train_args, angle, trans, back_angle, back_trans, scale_disp):
    
    loss = 0
    loss_records = 0
    smooth_losses = 0
    device = model_outs['pred_disp',0].device

    color = inputs['color',0,0]
    target = inputs['color',0,0]

    backproject_depth = utils.BackprojectDepth(train_args.batch_size, train_args.re_height, train_args.re_width)
    backproject_depth.to(device)
    project_3d = utils.Project3D(train_args.batch_size, train_args.re_height, train_args.re_width)
    project_3d.to(device)

    front_T = utils.transformation_from_parameters(angle[:, 0], trans[:, 0], invert=(-1<0))
    back_T = utils.transformation_from_parameters(back_angle[:,0],back_trans[:,0],invert=(1<0))

    for scale in range(4):
        
        reprojection_losses = []

        disp = model_outs['pred_disp',scale]
        ## resize disp to target_size
        disp = F.interpolate(disp, target.shape[-2:], mode="bilinear", align_corners = False)
        _, depth = utils.disp_to_depth(disp, train_args.min_depth, train_args.max_depth)
        
        model_outs['pred_depth',0, scale] = depth

        ## back to current
        cam_points = backproject_depth(depth, inputs['inv_K',0])
        pix_coords = project_3d(cam_points, inputs['K',0], front_T)

        front_repoj_image = F.grid_sample(inputs['color',-1,0],pix_coords.to(torch.float32),padding_mode="border")
        
        ## front to back
        if train_args.use_future_frame:
            cam_points = backproject_depth(depth, inputs['inv_K',0])
            pix_coords = project_3d(cam_points, inputs['K',0], back_T)

        back_repoj_image = F.grid_sample(inputs['color',1,0],pix_coords.to(torch.float32),padding_mode="border")
        
        if scale == 0:
            model_outs['reproj_img_from_prev'] = front_repoj_image
            model_outs['reproj_img_from_fut'] =  back_repoj_image
                   
        reprojection_losses.append(utils.compute_reprojection_loss(front_repoj_image, target))
        
        # JINLOVESPHO - future frame 도 사용
        if train_args.use_future_frame:
            reprojection_losses.append(utils.compute_reprojection_loss(back_repoj_image, color))

        reprojection_losses = torch.cat(reprojection_losses, 1)
        reprojection_loss = reprojection_losses

        # ## auto masking
        identity_reprojection_losses = []
        identity_reprojection_losses.append(
            utils.compute_reprojection_loss(inputs['color',-1,0], target))
        
        # JINLOVESPHO use future frame
        if train_args.use_future_frame:
            identity_reprojection_losses.append(
                utils.compute_reprojection_loss(inputs['color',1,0], target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
        identity_reprojection_loss = identity_reprojection_losses

        identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape, device=device) * 0.00001

        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

        mask = torch.argmin(combined, dim=1).unsqueeze(1).float()
        mask[mask>=1] = 1.0
        loss_record = mask * reprojection_loss / mask.sum().detach()
        loss_records += loss_record.mean().detach()

        to_optimise, idxs = torch.min(combined, dim=1)
        
        # JINLOVESPHO
        # disp1 = F.interpolate(model_outs1['pred_disp',scale], target.shape[-2:], mode="bilinear", align_corners = False)
        # _, depth1 = utils.disp_to_depth(disp1, train_args.min_depth, train_args.max_depth)
        moving_mask = torch.abs(model_outs['pred_depth',0, scale] - model_outs1['pred_depth',0,scale]) / model_outs1['pred_depth',0,scale]
        moving_mask = moving_mask < 0.1
        
        to_optimise = moving_mask.detach() * to_optimise
        
        if train_args.log_tool == 'wandb' and mode==EVAL: 
            msked_img = inputs['color',0,0][0].clone().permute(1,2,0).detach().cpu().numpy()
            wandb.log({"moving_mask": wandb.Image(moving_mask[0].detach().cpu().numpy()*100)})
            wandb.log({"moving_masked_img": wandb.Image(msked_img)})
        
        loss += 0.01*(1-moving_mask.detach().int())*torch.abs(model_outs['pred_depth',0,scale]- model_outs1['pred_depth',0,scale])
        
        # msk_in, msk_out, depth_diff_dict = utils.compute_twice_depth_loss(model_outs1['pred_depth',0,scale], model_outs['pred_depth',0,scale])        
        # to_optimise = msk_in.detach() * to_optimise
        
        # if train_args.log_tool == 'wandb' and mode == EVAL:
        #     wandb.log(depth_diff_dict)
        #     wandb.log({'auto_mask':wandb.Image(mask[0].detach().cpu().numpy()*100.0)})
        #     wandb.log({'mving_obj_msk':wandb.Image(msk_in[0].detach().cpu().numpy()*100.0)})
            
        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = utils.get_smooth_loss(norm_disp, target)

        smooth_losses += train_args.smooth_weight * smooth_loss / (2 ** scale)

    return loss/4.0, mask, loss_records, smooth_losses/4.0