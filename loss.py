import torch
import torch.nn.functional as F
import torchvision 
import utils
from utils import * 

import sys
import pdb

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
def compute_loss(inputs, model, train_args, mode = TRAIN):
    losses = {}
    total_loss = 0
    
    orig_h, orig_w = inputs['depth_gt'].shape[-2:]
    gt_depth = inputs['depth_gt']
    
    # forward pass 
    model_outs = model_forward(inputs, model, train_args, mode)  # (b,1,192,640)
    
    # supervised training
    if train_args.training_loss == 'supervised_depth':
        pred_depth_orig = F.interpolate(model_outs['pred_depth'], size=(orig_h, orig_w), mode="bilinear", align_corners = False)   # (b,1,375,1242)
        non_zero_mask = (inputs['depth_gt'] > 0).detach() 
        losses['sup_loss'] = compute_sup_loss(pred_depth_orig, gt_depth, non_zero_mask)


    # self-supervised training
    elif train_args.training_loss == 'selfsupervised_img_recon': 
 
        recon_losses = []
        smooth_losses = []

        #forward pose_net
        fa,ft,ba,bt = pose_forward(inputs, model)

        recon_loss, mask, _,smooth_loss = compute_selfsup_mono_loss(model_outs, inputs, train_args, fa, ft,ba,bt)
        recon_losses.append(recon_loss)
        smooth_losses.append(smooth_loss)
        
        pred_depth_orig = F.interpolate(model_outs['pred_depth',0,0], (orig_h, orig_w), mode="bilinear", align_corners = False)   # (b,1,375,1242)
        
        losses['selfsup_loss'] = torch.stack(recon_losses).mean()
        losses['smooth_loss'] = torch.stack(smooth_losses).mean()
        
    else:
        pass
        
        
        
    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    # returns
    if mode == TRAIN:
        return total_loss, losses
    else:
        return total_loss, losses, pred_depth_orig, model_outs


def model_forward(inputs, model, train_args, mode):
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

def compute_selfsup_mono_loss(model_outs, inputs, train_args, angle, trans, back_angle, back_trans):
# def compute_selfsup_mono_loss(label_pred_depth, label, train_args, angle, trans, back_angle, back_trans, scale_disp):
    
    loss = 0 
    loss_records = 0
    smooth_loss = 0
    device = model_outs['pred_disp',0].device

    color = inputs['color',0,0]
    target = inputs['color_aug',0,0]

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

        smooth_loss += (1e-3) * smooth_loss / (2 ** scale)

    return loss/4.0, mask, loss_records, smooth_loss/4.0