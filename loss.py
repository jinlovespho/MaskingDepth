import torch
import torch.nn.functional as F
import torchvision 
import utils
from utils import * 

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
    
    # loss calculation
    if train_args.training_loss == 'supervised_depth':
        pred_depth_orig = F.interpolate(model_outs['pred_depth'], (orig_h, orig_w), mode="bilinear", align_corners = False)   # (b,1,375,1242)
        non_zero_mask = (inputs['depth_gt'] > 0).detach() 
        losses['sup_loss'] = compute_sup_loss(pred_depth_orig, gt_depth, non_zero_mask)

    elif train_args.training_loss == 'selfsupervised_img_recon':    
        pred_depth_orig = F.interpolate(model_outs['~~'], (orig_h, orig_w), mode="bilinear", align_corners = False)   # (b,1,375,1242)
        recon_losses = []
        recon_loss, recon_img = compute_selfsup_loss(inputs, model_outs, train_args, mode)
        recon_losses.append(recon_loss)  
        losses['selfsup_loss'] = torch.stack(recon_losses).mean()


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



############################################################################## 
########################    loss function set
############################################################################## 
  
def compute_sup_loss(pred_depth, gt_depth, non_zero_mask):
    if non_zero_mask == None:
        loss = torch.abs(pred_depth - gt_depth.detach()).mean()
    else:
        loss = torch.abs(pred_depth[non_zero_mask] - gt_depth.detach()[non_zero_mask]).mean()
    return loss

def compute_selfsup_loss(inputs, model_outs, train_args, mode):
    loss=0
    loss_records=0
    
    device = model_outs['pred_depth4'].device
    
    axis_trans = model_outs['ca_map4_pose'].mean(dim=[2,3,4,5]).view(-1,2,1,6) # (b,12) -> (b,2,1,6)
    angle = axis_trans[..., :3]     # (b,2,1,3)
    translation = axis_trans[..., 3:]   # (b,2,1,3)
    rel_pose = transformation_from_parameters(angle[:, 0], translation[:, 0], invert=True)  # (b,4,4)
    
    backproject_depth = utils.BackprojectDepth(train_args.batch_size, train_args.re_height, train_args.re_width)
    backproject_depth.to(device)
    project_3d = utils.Project3D(train_args.batch_size, train_args.re_height, train_args.re_width)
    project_3d.to(device)
    
    
    reprojection_losses = []
    
    ## back to current
    cam_points = backproject_depth(model_outs['pred_depth4'], inputs['inv_K',0])
    pix_coords = project_3d(cam_points, inputs['K',0], rel_pose)

    recon_img = F.grid_sample(inputs['color',-1,0], pix_coords.to(torch.float32), padding_mode="border")
     
    reprojection_losses.append(utils.compute_reprojection_loss(recon_img, inputs['color',0,0]))
    reprojection_losses = torch.cat(reprojection_losses, 1)
    reprojection_loss = reprojection_losses
    
    # ## auto masking
    identity_reprojection_losses = []
    identity_reprojection_losses.append(
        utils.compute_reprojection_loss(inputs['color',-1,0], inputs['color',0,0]))
    
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
    
    return loss, recon_img

    # mean_disp = disp.mean(2, True).mean(3, True)
    # norm_disp = disp / (mean_disp + 1e-7)
    # smooth_loss = utils.get_smooth_loss(norm_disp, target)

    # smooth_loss += (1e-3) * smooth_loss / (2 ** scale)


def compute_sup_mask_loss(pred_depth, gt_depth): 
    pred_depth = F.interpolate(pred_depth, gt_depth.shape[-2:], mode="bilinear", align_corners = False)
    return utils.ssi_log_mask_loss(pred_depth+1, gt_depth.detach()+1, (gt_depth > 0).detach())

