import torch
import torch.nn.functional as F
import torchvision 
import utils

TRAIN   = 0
EVAL    = 1

# total loss(sup)
def compute_loss(inputs, model, train_cfg, mode = TRAIN):
    losses = {}
    total_loss = 0
    
    # breakpoint()

    pred_depth, full_features, fusion_features = model_forward(inputs['color'], model)
    
    # breakpoint()

    ### compute supervised loss 
    if train_cfg.data.dataset in ['nyu'] or train_cfg.unlabeled_data.dataset in ['nyu']:
        losses['sup_loss'] = compute_sup_loss(pred_depth, inputs['depth_gt'])
    # 여기 실행 for KITTI 
    else:
        pred_depth = F.interpolate(pred_depth, inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)   # model의 output인 pred_depth를 gt_depth_map 크기로 interpolate하여 scale 맞춘것
        losses['sup_loss'] = compute_sup_loss(pred_depth, inputs['depth_gt'], (inputs['depth_gt'] > 0).detach())        # scale 맞춘 pred_depth 와 g.t_depth 실제 loss 계산 
    
    ### make uncertainty map
    if 'uncert_decoder' in model.keys():
        pred_uncert = uncert_forward(fusion_features, model)
        if train_cfg.data.dataset == 'nyu' or train_cfg.unlabeled_data.dataset == 'nyu':
            losses['uncert_loss'] = compute_uncert_loss(pred_uncert, pred_depth, inputs['depth_gt'])
        else:
            pred_uncert = F.interpolate(pred_uncert, inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)
            losses['uncert_loss'] = compute_uncert_loss(pred_uncert, pred_depth, inputs['depth_gt'], (inputs['depth_gt'] > 0).detach())
    else: 
        pred_uncert = None 

    ### compute consistency loss
    if not(train_cfg.d_consistency == 0) or not(train_cfg.f_consistency == 0):
        
        #### make K-way augmented depth map
        pred_depth_mask, mask_features, _, = model_forward(inputs['color_aug'], model, K = train_cfg.K)

        if not(train_cfg.data.dataset == 'nyu' or train_cfg.unlabeled_data.dataset == 'nyu'):
            pred_depth_mask = F.interpolate(pred_depth_mask, inputs['depth_gt'].shape[-2:], mode="bilinear", align_corners = False)

        ### consistency loss between weak depth map and strong depth map
        if not(train_cfg.d_consistency == 0):
            losses['consistency_loss'] = train_cfg.d_consistency * compute_adaptive_consistency_loss(pred_depth, pred_depth_mask, pred_uncert)
        if not(train_cfg.f_consistency == 0):
            losses['feature_consistency_loss'] = train_cfg.f_consistency * compute_feature_consistency_loss(full_features, mask_features, model)
        
    else:
        if mode == EVAL:
            pred_depth_mask, _, _ = model_forward(inputs['color_aug'], model, K = train_cfg.K)
        else:
            pred_depth_mask = None


    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    if mode == TRAIN:
        return total_loss, losses
    else:
        return total_loss, losses, pred_depth, pred_uncert, pred_depth_mask


def compute_loss_multiframe_colorLoss(inputs, model, train_cfg, mode = TRAIN):
    losses = {}
    total_loss = 0
    
    # list 형태로 정리된 time frame들을 dic 형태로 다시 구성
    inputs_dic={}
    key_lst = [ key for key in inputs[0].keys() ]
    for key in key_lst:
        inputs_dic[key]=[]
    for input in inputs:
        for key, val in input.items():
           inputs_dic[key].append(val)
    
    # breakpoint()
            
    pred_depth, pred_color, full_features, fusion_features = model_forward_multiframe_colorLoss(inputs_dic['color'], model, train_cfg.K,  mode)
    
    reconstruction_size = inputs_dic['depth_gt'][0].shape[-2:]  # (375,1242)
    pred_depth = F.interpolate(pred_depth, reconstruction_size, mode="bilinear", align_corners = False)   # model의 output인 pred_depth를 gt_depth_map 크기로 interpolate하여 scale 맞춘것
    losses['sup_loss_depth'] = compute_sup_loss(pred_depth, inputs_dic['depth_gt'][0], (inputs_dic['depth_gt'][0] > 0).detach())        # scale 맞춘 pred_depth 와 g.t_depth 실제 loss 계산 
    losses['sup_loss_color'] = compute_sup_loss(pred_color, inputs_dic['color'][0], mask=None)
        
    ### make uncertainty map
    pred_uncert = None 


    if mode == EVAL:
        pred_depth_mask, pred_color_mask, _, _ = model_forward_multiframe_colorLoss(inputs_dic['color_aug'], model, K = train_cfg.K)
    else:
        pred_depth_mask = None


    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    if mode == TRAIN:
        return total_loss, losses
    else:
        return total_loss, losses, pred_depth, pred_color, pred_uncert, pred_depth_mask


# JINLOVESPHO
def compute_loss_multiframe(inputs, model, train_cfg, mode = TRAIN):
    losses = {}
    total_loss = 0
    
    # list 형태로 정리된 time frame들을 dic 형태로 다시 구성
    inputs_dic={}
    key_lst = [ key for key in inputs[0].keys() ]
    for key in key_lst:
        inputs_dic[key]=[]
    for input in inputs:
        for key, val in input.items():
           inputs_dic[key].append(val)

    
    pred_depth, full_features, fusion_features = model_forward_multiframe(inputs_dic['color'], model, train_cfg.K,  mode)

    # breakpoint()
    pred_depth = F.interpolate(pred_depth, inputs_dic['depth_gt'][0].shape[-2:], mode="bilinear", align_corners = False)   # model의 output인 pred_depth를 gt_depth_map 크기로 interpolate하여 scale 맞춘것
    losses['sup_loss'] = compute_sup_loss(pred_depth, inputs_dic['depth_gt'][0], (inputs_dic['depth_gt'][0] > 0).detach())        # scale 맞춘 pred_depth 와 g.t_depth 실제 loss 계산 
        
    pred_uncert = None 

    if mode == EVAL:
        pred_depth_mask, _, _ = model_forward_multiframe(inputs_dic['color_aug'], model, K = train_cfg.K)
    else:
        pred_depth_mask = None


    #total_loss
    for loss in losses.values():
        total_loss += loss
    
    if mode == TRAIN:
        return total_loss, losses
    else:
        return total_loss, losses, pred_depth, pred_uncert, pred_depth_mask



# main network forward
def model_forward(inputs, model, K=1):  
    pred_depth, features, fusion_features = model['depth'](inputs, K)
    return pred_depth, features, fusion_features

# JINLOVESPHO
def model_forward_multiframe_colorLoss(inputs, model, K=1, mode=None):  
    pred_depth, pred_color, features, fusion_features = model['depth'](inputs, K, mode)
    return pred_depth, pred_color, features, fusion_features

# JINLOVESPHO
def model_forward_multiframe(inputs, model, K=1, mode=None):  
    pred_depth, features, fusion_features = model['depth'](inputs, K, mode)
    return pred_depth, features, fusion_features


############################################################################## 
########################    loss function set
############################################################################## 
  
def compute_sup_loss(pred_depth, gt_depth, mask=None): 
    # breakpoint()
    if mask == None:
        loss = torch.abs(pred_depth - gt_depth.detach()).mean()
    else:
        loss = torch.abs(pred_depth[mask] - gt_depth.detach()[mask]).mean()
    return loss

def compute_sup_mask_loss(pred_depth, gt_depth): 
    pred_depth = F.interpolate(pred_depth, gt_depth.shape[-2:], mode="bilinear", align_corners = False)
    return utils.ssi_log_mask_loss(pred_depth+1, gt_depth.detach()+1, (gt_depth > 0).detach())

