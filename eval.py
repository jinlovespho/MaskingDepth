import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import PIL.Image as pil 
import cv2 
import utils
import wandb

def evaluate_metric(train_cfg, pred_depth, inputs):
    if train_cfg.real.dataset ==  'cityscape':
        metric = eval_cityscape(pred_depth, inputs)
    elif train_cfg.real.dataset == 'nyu':
        metric = eval_nyu(pred_depth, inputs)
    elif train_cfg.real.dataset == 'virtual_kitti':
        metric = eval_virtual_kitti(pred_depth, inputs)
    else:
        metric = eval_kitti(pred_depth, inputs)

    return metric

def eval_kitti(pred_depth, inputs):
    pred_depth = torch.clamp(F.interpolate(pred_depth, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_gt = inputs["depth_gt"]
    mask = depth_gt > 0
    crop_mask = torch.zeros_like(mask)
    crop_mask[:, :, 153:371, 44:1197] = 1
    mask = mask * crop_mask

    depth_gt = depth_gt[mask]
    pred_depth = pred_depth[mask]

    pred_depth[pred_depth < 1e-03] = 1e-03
    pred_depth[pred_depth > 80] = 80

    pred_depth *= torch.median(depth_gt) / torch.median(pred_depth)
    pred_depth = torch.clamp(pred_depth, min=1e-03, max=80)
    depth_errors = [*utils.compute_depth_errors(depth_gt, pred_depth)]

    return depth_errors

def eval_cityscape(pred_depth, inputs):
    return [0, 0, 0, 0, 0, 0, 0]

def eval_nyu(pred_depth, inputs):
    pred_depth  = pred_depth[:,:,45:471, 41:601]
    depth_gt    = inputs["depth_gt"][:,:,45:471, 41:601]

    pred_depth[pred_depth < 1e-03] = 1e-03
    pred_depth[pred_depth > 10] = 10

    pred_depth *= torch.median(depth_gt) / torch.median(pred_depth)
    pred_depth = torch.clamp(pred_depth, min=1e-03, max=10)
    depth_errors = [*utils.compute_depth_errors(depth_gt, pred_depth)]
    return depth_errors

def eval_virtual_kitti(pred_depth, inputs):
    pred_depth = utils.disp_to_depth(pred_depth, 1e-3, 80)[-1]
    pred_depth = torch.clamp(F.interpolate(pred_depth, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
    depth_gt = inputs["depth_gt"]
    
    pred_depth[pred_depth < 1e-3]  = 1e-3
    pred_depth[pred_depth > 80]    = 80

    pred_depth *= torch.median(depth_gt) / torch.median(pred_depth)
    pred_depth = torch.clamp(pred_depth, min=1e-3, max=80)
    depth_errors = [*utils.compute_depth_errors(depth_gt, pred_depth)]
    return depth_errors

def get_eval_dict(errors):
    mean_errors = np.array(errors).mean(1)
    depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
    error_dict = {}
    for error_name, error_value in zip(depth_metric_names, mean_errors):
        error_dict[error_name] = error_value.item()
    return error_dict

def eval_metric(pred_depths, gt_depths, train_args):
    
    # pred_depths [ (375,1242) . . . ]
    # gt_depths   [ (375,1242) . . . ]
    
    num_samples = len(pred_depths)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    ratios=[]
    
    for i in range(num_samples):
        # gt_depth and pred_depth are numpys
        
        MIN_DEPTH = 0.001
        MAX_DEPTH = (10.0  if train_args.dataset == 'nyu' else 80.0)  # 80
        
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape
        
        pred_depth = pred_depths[i]
         
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)
        
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        
        ratio = np.median(gt_depth) / np.median(pred_depth)
        ratios.append(ratio)
        pred_depth *= ratio
        
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth, pred_depth)
        
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    return abs_rel, sq_rel, rms, log_rms, d1, d2, d3


# def eval_metric(pred_depths, gt_depths, train_args):
    
#     # pred_depths [ (375,1242) . . . ]
#     # gt_depths   [ (375,1242) . . . ]
    
#     num_samples = len(pred_depths)

#     silog = np.zeros(num_samples, np.float32)
#     log10 = np.zeros(num_samples, np.float32)
#     rms = np.zeros(num_samples, np.float32)
#     log_rms = np.zeros(num_samples, np.float32)
#     abs_rel = np.zeros(num_samples, np.float32)
#     sq_rel = np.zeros(num_samples, np.float32)
#     d1 = np.zeros(num_samples, np.float32)
#     d2 = np.zeros(num_samples, np.float32)
#     d3 = np.zeros(num_samples, np.float32)
    
#     for i in range(num_samples):
#         # gt_depth and pred_depth are numpys
        
#         gt_depth = gt_depths[i]
#         pred_depth = pred_depths[i]
     
#         min_depth = 0.001
#         max_depth = (10.0  if train_args.dataset == 'nyu' else 80.0)  # 80
        
#         # clamp pred_depth values to min_depth~max_depth
#         pred_depth[pred_depth < min_depth] = min_depth
#         pred_depth[pred_depth > max_depth] = max_depth
#         pred_depth[np.isinf(pred_depth)] = max_depth
        
#         gt_depth[np.isinf(gt_depth)] = 0
#         gt_depth[np.isnan(gt_depth)] = 0

#         valid_mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)     # (375,1242) : true/false mask

#         gt_height, gt_width = gt_depth.shape
#         eval_mask = np.zeros(valid_mask.shape)  # (375,1242) filled with zeros

#         if train_args.dataset == 'nyu':
#             eval_mask[45:471, 41:601] = 1
#         else:
#             # eval_mask[153:371, 44:1197]=255
#             # cv2.imwrite('../eval_mask_region.png', eval_mask)
            
#             # kitti에서 eval 할 영역 지정
#             eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1   # eval_mask[153:371, 44:1197]

#         # final mask for calculating only on valid regions
#         valid_mask = np.logical_and(valid_mask, eval_mask)
        
#         valid_gt_depth = gt_depth[valid_mask]
#         valid_pred_depth = pred_depth[valid_mask]
        
#         if train_args.training_loss == 'selfsupervised_img_recon':
#             valid_pred_depth *= np.median(valid_gt_depth) / np.median(valid_pred_depth)
#             # pred_depth *= np.median(gt_depth) / np.median(pred_depth) 확실히 valid_pred_depth랑 차이가 있네

#         silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(valid_gt_depth, valid_pred_depth)
    
#     print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
#         'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
#     print("{:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}".format(
#         d1.mean(), d2.mean(), d3.mean(),
#         abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

#     return abs_rel, sq_rel, rms, log_rms, d1, d2, d3




def eval_metric_bbox(pred_depths, gt_depths, data, bbox_mask_depths):
    
    # pred_depths [ (375,1242) . . . ]
    # gt_depths   [ (375,1242) . . . ]
    
    num_samples = len(pred_depths)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)
    
    for i in range(num_samples):
        

        gt_depth = gt_depths[i]
        pred_depth = pred_depths[i]
        bbox_mask_depth = bbox_mask_depths[i]
        
        min_depth = 0.001
        max_depth = (10.0  if data.dataset == 'nyu' else 80.0)  # 80

        # clamp values of pred_depths to min_depth and max_depth
        # predicted depth value 가 [min_depth, max_depth] 범위를 갖도록 조정
        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        pred_depth[np.isinf(pred_depth)] = max_depth
        
        gt_depth[np.isinf(gt_depth)] = 0
        gt_depth[np.isnan(gt_depth)] = 0

        valid_mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)     # (375,1242) : true/false mask

        gt_height, gt_width = gt_depth.shape
        eval_mask = np.zeros(valid_mask.shape)  # (375,1242) filled with zeros

        if data.dataset == 'nyu':
            eval_mask[45:471, 41:601] = 1
        else:
            # eval_mask[153:371, 44:1197]=255
            # cv2.imwrite('../eval_mask_region.png', eval_mask)
            
            # kitti에서 eval 할 영역 지정
            eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1   # eval_mask[153:371, 44:1197]

        # 최종적으로 valid 한 영역만 계산 하는 mask
        valid_mask = np.logical_and(valid_mask, eval_mask)
        
        # 추가로 bbox 영역만 계산하도록 mask 설정
        valid_mask_bbox = np.logical_and(valid_mask, bbox_mask_depth)
        # print( valid_mask.size - np.count_nonzero(valid_mask) )   # number of zeros in array

        # tmp2 = np.uint8(valid_mask)*255
        # tmp3 = np.uint8(valid_mask_bbox)*255
        
        # cv2.imwrite('../tmptmp2.png', tmp2)
        # cv2.imwrite('../tmptmp3.png', tmp3)
    
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(gt_depth[valid_mask_bbox], pred_depth[valid_mask_bbox])
    
    # nan 처리 mask 생성
    nan_msk_silog = np.isnan(silog)
    nan_msk_log10 = np.isnan(log10)
    nan_msk_rms = np.isnan(rms)
    nan_msk_log_rms = np.isnan(log_rms)
    nan_msk_abs_rel = np.isnan(abs_rel)
    nan_msk_sq_rel = np.isnan(sq_rel)
    nan_msk_d1 = np.isnan(d1)
    nan_msk_d2 = np.isnan(d2)
    nan_msk_d3 = np.isnan(d3)
    
    # nan 제외한 값들만 남기기
    silog = silog[~nan_msk_silog]
    log10 = log10[~nan_msk_log10]
    rms = rms[~nan_msk_rms]
    log_rms = log_rms[~nan_msk_log_rms]
    abs_rel = abs_rel[~nan_msk_abs_rel]
    sq_rel = sq_rel[~nan_msk_sq_rel]
    d1 = d1[~nan_msk_d1]
    d2 = d2[~nan_msk_d2]
    d3 = d3[~nan_msk_d3]
    
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}, {:7.4f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    return abs_rel, sq_rel, rms, log_rms, d1, d2, d3


def compute_errors(gt, pred):
    
    thresh = np.maximum((gt / pred), (pred / gt))   # np.array 꼴의 thresh에 nan 이 하나라도 있으면, thresh.mean() 할 경우 nan이 뜬다.
    
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


# visualize on wandb
def visualize(inputs, pred_depth, model_outs, train_args, sample_num=4):
    b = pred_depth.shape[0]
    sample_num = b if b < sample_num else sample_num
    
    orig_h, orig_w = inputs['depth_gt'].shape[-2:]
       
    input_curr_img = F.interpolate(inputs['color',0,0], size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    
    for i in range(sample_num):
        wandb_eval_dict = {}
        vis1 = []
        vis2 = []

        #rgb image 
        vis_curr_img = input_curr_img[i]
        vis_curr_img *= 255
        vis1.append(wandb.Image(vis_curr_img, caption="Curr Input Image"))
        
        # pred depth
        vis_pred_depth = pred_depth[i].squeeze().cpu().numpy() 
        vmax = np.percentile(vis_pred_depth, 95)
        normalizer = mpl.colors.Normalize(vmin=vis_pred_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_pred_depth = (mapper.to_rgba(vis_pred_depth)[:, :, :3] * 255).astype(np.uint8)
        vis_pred_depth = pil.fromarray(colormapped_pred_depth)
        vis1.append(wandb.Image(vis_pred_depth, caption="Pred Depth"))
        
        # gt_depth
        gt_depth = inputs['depth_gt'][i].squeeze().cpu().numpy() 
        vmax = np.percentile(gt_depth, 95)
        normalizer = mpl.colors.Normalize(vmin=gt_depth.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_gt_depth = (mapper.to_rgba(gt_depth)[:, :, :3] * 255).astype(np.uint8)
        gt_depth = pil.fromarray(colormapped_gt_depth)
        vis1.append(wandb.Image(gt_depth, caption="GT Depth"))
  
        
        if train_args.training_loss == 'selfsupervised_img_recon':
            
            # resize to orig size
            vis_curr_from_prev = F.interpolate(model_outs['reproj_img_from_prev'], size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            vis_curr_from_fut  = F.interpolate(model_outs['reproj_img_from_fut'], size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            # curr_img from prev_img 
            vis_curr_from_prev = vis_curr_from_prev[i]
            vis_curr_from_prev *= 255
            vis2.append(wandb.Image(vis_curr_from_prev, caption="Curr from Prev"))
            
            # curr_img from future_img
            vis_curr_from_fut = vis_curr_from_fut[i]
            vis_curr_from_fut *= 255
            vis2.append(wandb.Image(vis_curr_from_fut, caption="Curr from Fut"))
            
            
        wandb_eval_dict['vis1'] = vis1
        wandb_eval_dict['vis2'] = vis2
        
        wandb.log(wandb_eval_dict)