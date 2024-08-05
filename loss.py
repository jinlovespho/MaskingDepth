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
def compute_loss(inputs, model, train_args, mode = TRAIN,epoch=0):
    losses = {}
    total_loss = 0
    
    orig_h, orig_w = inputs['depth_gt'].shape[-2:]
    gt_depth = inputs['depth_gt']
    
    # forward pass 
    if train_args.with_pose:
        model_outs, model_outs_back = model_forward(inputs, model, train_args, mode, train_args.with_pose)  # (b,1,192,640)
    else:
        model_outs = model_forward(inputs, model, train_args, mode, train_args.with_pose)  # (b,1,192,640)
        
    if 'no_grad' in train_args.moving_masking:
        with torch.no_grad():
            input_tt = {('color_aug',0,0):inputs['color_aug',0,0].clone(), ('color_aug',-1,0):inputs['color_aug',0,0].clone(), ('K',0):inputs['K',0].clone(),\
                        ('color',0,0):inputs['color',0,0].clone(), ('color',-1,0):inputs['color',0,0].clone(), ('inv_K',0):inputs['inv_K',0].clone()}
            model_outs_tt = model_forward(input_tt, model, train_args, mode, train_args.with_pose)
    else:
        model_outs_tt = None
        
    # supervised training
    if train_args.training_loss == 'supervised_depth':
        # breakpoint()
        pred_depth_orig = F.interpolate(model_outs['pred_depth'], size=(orig_h, orig_w), mode="bilinear", align_corners = True)   # (b,1,375,1242)
        non_zero_mask = (inputs['depth_gt'] > 0).detach() 
        losses['sup_loss'] = compute_sup_loss(pred_depth_orig, gt_depth, non_zero_mask)


    # self-supervised training
    elif train_args.training_loss == 'selfsupervised_img_recon': 
 
        recon_losses = []
        smooth_losses = []

        #forward pose_net
        if not train_args.with_pose:
            fa,ft,ba,bt = pose_forward(inputs, model)
            front_pose, back_pose = None, None
        else:
            front_pose = model_outs['pose']
            back_pose = model_outs_back['pose']
            fa,ft,ba,bt = None, None, None, None
            
            for i in range(4):
                model_outs['pred_disp',i] = (model_outs['pred_disp',i]+model_outs_back['pred_disp',i])/2.
            

        recon_loss, mask, _,smooth_loss = compute_selfsup_mono_loss(model_outs, inputs, train_args, fa, ft,ba,bt, front_pose=front_pose, back_pose=back_pose, model_outs_tt=model_outs_tt,mode=mode,epoch=epoch)
        recon_losses.append(recon_loss)
        smooth_losses.append(smooth_loss)
        
        model_outs['pose_tmp'] = utils.transformation_from_parameters(fa[:, 0], ft[:, 0], invert=(-1<0))
        
        pred_depth_orig = F.interpolate(model_outs['pred_depth',0,0], (orig_h, orig_w), mode="bilinear", align_corners = True)   # (b,1,375,1242)
        
        losses['selfsup_loss'] = torch.stack(recon_losses).mean()
        losses['smooth_loss'] = torch.stack(smooth_losses).mean()
        
    else:
        pass
        
        
        
    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    # returns
    if mode == TRAIN:
        return total_loss, losses, model_outs
    else:
        return total_loss, losses, pred_depth_orig, model_outs


def model_forward(inputs, model, train_args, mode, with_pose = False):
    if with_pose:
        outputs = model['depth'](inputs[('color',0,0)], inputs[('color',-1,0)], mode, intrinsics=inputs['K',0])      
        outputs_back = model['depth'](inputs[('color',0,0)], inputs[('color',1,0)], mode, intrinsics=inputs['K',0])
        
        return outputs, outputs_back
    
    inputs['tt_aug'] = torch.zeros(inputs['color_aug',0,0].shape[0])
    
    if train_args.model_info == 'croco':
        if mode == TRAIN:
            target = inputs['color_aug',0,0].clone()
            source = inputs['color_aug',-1,0].clone()
            for batch in range(source.shape[0]):
                rand_num = random.random()
                if rand_num < train_args.zero_aug:
                    source[batch] = target[batch]
                    inputs['tt_aug'][batch] = 1
                    
            outputs = model['depth'](target, source, mode, intrinsics=inputs['K',0])
        else:
            outputs = model['depth'](inputs[('color',0,0)], inputs[('color',-1,0)], mode, intrinsics=inputs['K',0])      
    else:
        outputs = model['depth'](inputs, train_args, mode)
    return outputs

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

def compute_selfsup_mono_loss(model_outs, inputs, train_args, angle, trans, back_angle, back_trans, front_pose=None, back_pose=None, model_outs_tt=None,mode=TRAIN, epoch=0):
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

    if not train_args.with_pose:
        front_T = utils.transformation_from_parameters(angle[:, 0], trans[:, 0], invert=(-1<0))
        back_T = utils.transformation_from_parameters(back_angle[:,0],back_trans[:,0],invert=(1<0))
    else:
        front_T = front_pose
        back_T = back_pose

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
        
        if train_args.moving_masking == 'no_grad':
            disp = F.interpolate(model_outs_tt['pred_disp',scale], target.shape[-2:], mode="bilinear", align_corners = False)
            _, depth = utils.disp_to_depth(disp, train_args.min_depth, train_args.max_depth)
            moving_mask = torch.abs(model_outs['pred_depth',0, scale] - depth) / depth
            moving_mask = moving_mask < train_args.masking_threshold
            
            to_optimise = moving_mask.detach() * to_optimise
            
            # if train_args.log_tool == 'wandb' and mode==EVAL: 
            #     wandb.log({"moving_mask": wandb.Image(moving_mask[0].detach().cpu().numpy()*100)})

        if train_args.moving_masking == 'no_grad_topk':
            disp = F.interpolate(model_outs_tt['pred_disp',scale], target.shape[-2:], mode="bilinear", align_corners = False)
            _, depth = utils.disp_to_depth(disp, train_args.min_depth, train_args.max_depth)
            abs_error = torch.abs(model_outs['pred_depth',0, scale] - depth) / depth

            ## mask out top k
            B,C,H,W = abs_error.shape
            moving_masks = torch.ones_like(abs_error)
            for i in range(B):
                moving_mask = abs_error[i].view(-1)
                topk = int(moving_mask.shape[0]*0.2)
                _, idx = torch.topk(moving_mask, topk)
                moving_mask = torch.ones_like(moving_mask)
                if inputs['tt_aug'][i] == 0:
                    moving_mask[idx] = 0
                moving_masks[i] = moving_mask.view(C,H,W)

            
            to_optimise = moving_masks.detach() * to_optimise
            
            # if train_args.log_tool == 'wandb' and mode==EVAL: 
            #     wandb.log({"moving_mask": wandb.Image(moving_masks[0].detach().cpu().numpy()*100)})
                
        elif train_args.moving_masking == 'no_grad_distill' and epoch>=1:
            disp = F.interpolate(model_outs_tt['pred_disp',scale], target.shape[-2:], mode="bilinear", align_corners = False)
            _, depth = utils.disp_to_depth(disp, train_args.min_depth, train_args.max_depth)
            moving_mask = torch.abs(model_outs['pred_depth',0, scale] - depth) / depth
            moving_mask = moving_mask < train_args.masking_threshold
            
            to_optimise = moving_mask.detach() * to_optimise
            
            if train_args.log_tool == 'wandb' and mode==EVAL: 
                wandb.log({"moving_mask": wandb.Image(moving_mask[0].detach().cpu().numpy()*100)})

            loss += 0.01*(1-moving_mask.detach().int())*torch.abs(model_outs['pred_depth',0,scale]- depth)


        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = utils.get_smooth_loss(norm_disp, target)

        smooth_losses += train_args.smooth_weight * smooth_loss / (2 ** scale)

    return loss/4.0, mask, loss_records, smooth_losses/4.0