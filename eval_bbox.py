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
from eval import visualize, eval_metric, get_eval_dict, eval_metric_bbox

from torchvision.utils import save_image
import torch.nn as nn
import numpy as np
import cv2 

from vit_rollout import rollout
import torch.nn.functional as F
from torch.utils.data import DataLoader

from noise import image_corruption

TRAIN = 0
EVAL  = 1

parser = argparse.ArgumentParser(description='arguments yaml load')
parser.add_argument("--conf",
                    type=str,
                    help="configuration file path",
                    default="./conf/base_train.yaml")

args = parser.parse_args()

def return_mask_tensor_numpy(tkn):
    # kitti
    patch_num_h=12
    patch_num_w=40
    resized_h = 192
    resized_w = 640
    orig_h=375
    orig_w=1242
    
    # breakpoint()
    
    tkn = tkn.reshape(patch_num_h, patch_num_w).numpy()
    tkn = tkn / tkn.max()
    mask = cv2.resize(tkn, (resized_w, resized_h) )
    
    return mask
 
    tkn = tkn.view(patch_num_h, patch_num_w)    # (12,40)
    
    tkn = tkn / tkn.max()
    tkn = tkn[None,None,...]
    vis = nn.functional.interpolate(tkn, size=(resized_h, resized_w), mode='bilinear')
    vis_t = (vis-vis.min())/(vis.max()-vis.min())
    
    vis_n = vis_t.numpy() * 255 
    vis_n = np.uint8(vis_n)
    
    return vis_t, vis_n
       
def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def pho_visualize_cross_attn_map(ca_module_num, iter_num, curr_frame, prev_frame, ca_maps, discard_ratio, vis_idx, save_folder_name):

    curr_frame_n = curr_frame[0,...].permute(1,2,0).detach().cpu().numpy() *255    # (192,640,3) [0 255] 
    prev_frame_n = prev_frame[0,...].permute(1,2,0).detach().cpu().numpy() *255     # (192,640,3) [0 255]
    
    curr_frame_n = curr_frame_n.astype(np.uint8)
    prev_frame_n = prev_frame_n.astype(np.uint8)    
        
    curr_frame_n = curr_frame_n[:,:,[2,1,0]]
    prev_frame_n = prev_frame_n[:,:,[2,1,0]]
    
    # breakpoint()
    ca_map = rollout(ca_maps, discard_ratio=discard_ratio, head_fusion='mean')    # (n,n) = (480,480)
    # ca_map = torch.cat(ca_maps, dim=1)    
    # ca_map = ca_map.mean(dim=(0,1))   # (480,480)

    # token to visualize ca_map
    to_vis_tkn = ca_map[vis_idx,:]  # (480)
    h_idx = vis_idx//40
    w_idx = vis_idx%40
    
    # show ca_map
    mask= return_mask_tensor_numpy(to_vis_tkn)  # (192,640)
    result = show_mask_on_image(prev_frame_n, mask) # (192,640,3)
    
    # plot query point for curr_frame
    query=torch.zeros(12,40)
    query[h_idx,w_idx]=1
    query = nn.functional.interpolate(query[None,None,...], size=(192,640), mode='bilinear')
    
    query_numpy = query[0,0,...].detach().cpu().numpy()    # (192,640)

    # show query_point on curr_frame
    query_result = show_mask_on_image(curr_frame_n, query_numpy) # (192,640,3)

    # make dir for ca_maps
    save_path_ca_map=f'../vis_bbox/{save_folder_name}/ca_maps'
    if not os.path.exists(save_path_ca_map):
        os.makedirs(save_path_ca_map)

    if iter_num == 0 and ca_module_num == 1:
        cv2.imwrite(f'{save_path_ca_map}/vis_idx{vis_idx}_ca_map_curr.jpg', query_result)
    cv2.imwrite(f'{save_path_ca_map}/vis_idx{vis_idx}_module{ca_module_num}_ca_map_prev.jpg', result )

# Define a function to draw bounding boxes on an image
def draw_boxes(image_tensor, boxes, color=(0, 255, 0), thickness=1):
    
    coco_label_names=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    
    images = []
    masks=[]
    tmp1=[]
    tmp2=[]
    for i, img_tensor in enumerate(image_tensor):
        image = img_tensor.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype(np.uint8).copy()  # Convert to uint8
        mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create an empty mask

        for box in boxes[i]:
            cls, h1, w1, h2, w2, confidence = box.tolist()
            if cls > 0:   
                x1, y1, x2, y2 = int(h1), int(w1), int(h2), int(w2)

                # bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                label_name = coco_label_names[int(cls)-1]
                label = f"{label_name}"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

                # mask
                cv2.rectangle(mask, (x1, y1), (x2, y2), (1), thickness=-1)  # Fill the box region with ones in the mask
        
        images.append(torch.tensor(image).permute(2, 0, 1).float())
        masks.append(torch.tensor(mask).unsqueeze(0).float())
    #     tmp1.append(torch.tensor(image).permute(2, 0, 1).float())
    #     tmp2.append(torch.tensor(mask).unsqueeze(0).float())
        
    #     save_image(tmp1[0].float(), f'../tmp1.png', normalize=True)
    #     save_image(tmp2[0].float(), f'../tmp2.png', normalize=True)

    # breakpoint()
    
    return images,masks    



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
        train_ds, val_ds, train_loader, val_loader = initialize.data_loader(train_cfg.data, train_cfg.batch_size, train_cfg.num_workers)
                                        
        # set wandb
        if train_cfg.wandb:
            wandb.init(project = train_cfg.wandb_proj_name,
                        name = train_cfg.model_name,
                        config = conf,
                        dir=train_cfg.wandb_log_path)
        # save configuration (this part activated when do not use wandb)
        else: 
            save_config_folder = os.path.join(train_cfg.log_path, train_cfg.model_name)
            if not os.path.exists(save_config_folder):
                os.makedirs(save_config_folder)
            with open(save_config_folder + '/config.yaml', 'w') as f:
                yaml.dump(conf, f)
            progress = open(save_config_folder + '/progress.txt', 'w')
                
        step = 0    
        #validation
        with torch.no_grad():
            
            MIN_DEPTH = 1e-3
            MAX_DEPTH = 80

            frames_to_load = [0]
            for idx in range(-1, -1 - 1, -1):
                if idx not in frames_to_load:
                    frames_to_load.append(idx)

            if True:
                from networks.manydepth.resnet_encoder import ResnetEncoderMatching
                
                load_weights_folder = '/home/kwangrok/Downloads/VS_CODE/my_github/monodepth/anhg/MaskingDepth/pretrained_weights/KITTI_MR'
                encoder_path = os.path.join(load_weights_folder, "encoder.pth")
                decoder_path = os.path.join(load_weights_folder, "depth.pth")
                encoder_class = ResnetEncoderMatching

                encoder_dict = torch.load(encoder_path)
                
                try:
                    HEIGHT, WIDTH = encoder_dict['height'], encoder_dict['width']
                except KeyError:
                    print('No "height" or "width" keys found in the encoder state_dict, resorting to '
                        'using command line values!')
                    HEIGHT, WIDTH = 192, 640

                # encoder config 느낌 
                encoder_opts = dict(num_layers=18,
                                    pretrained=False,
                                    input_width=encoder_dict['width'],
                                    input_height=encoder_dict['height'],
                                    adaptive_bins=True,
                                    min_depth_bin=0.1, max_depth_bin=20.0,
                                    depth_binning='linear',
                                    num_depth_bins=96)
                pose_enc_dict = torch.load(os.path.join(load_weights_folder, "pose_encoder.pth"))
                pose_dec_dict = torch.load(os.path.join(load_weights_folder, "pose.pth"))

                # 여기가 manydepth fig2. 의 하나의 poseCNN 이다.
                from networks.manydepth import resnet_encoder, pose_decoder
                pose_enc = resnet_encoder.ResnetEncoder(18, False, num_input_images=2)
                pose_dec = pose_decoder.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)

                enc_weights=pose_enc.load_state_dict(pose_enc_dict, strict=True)
                dec_weights=pose_dec.load_state_dict(pose_dec_dict, strict=True)
                print(enc_weights)
                print(dec_weights)

                min_depth_bin = encoder_dict.get('min_depth_bin')
                max_depth_bin = encoder_dict.get('max_depth_bin')

                pose_enc.eval()
                pose_dec.eval()

                if torch.cuda.is_available():
                    pose_enc.cuda()
                    pose_dec.cuda()
                    
                from networks.manydepth.depth_decoder import DepthDecoder
                encoder = encoder_class(**encoder_opts) # ResnetEncoderMatching 이라는 class 
                depth_decoder = DepthDecoder(encoder.num_ch_enc)

                model_dict = encoder.state_dict()
                enc_load_result = encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
                dec_load_result = depth_decoder.load_state_dict(torch.load(decoder_path))

                print(enc_load_result)
                print(dec_load_result)

                encoder.eval()
                depth_decoder.eval()

                if torch.cuda.is_available():
                    encoder.cuda()
                    depth_decoder.cuda()

                pred_disps = []

                print("-> Computing predictions with size {}x{}".format(HEIGHT, WIDTH))
            
            # breakpoint()
              
            # MY
            utils.model_mode(model,EVAL)
              
            # LOAD MODEL and VISUALIZE ATTENTION MAPS
            load_path = train_cfg.model.my_load_path
            loaded_weight = torch.load(load_path)
            load_result = model['depth'].module.load_state_dict(loaded_weight, strict=False)
            print(load_result)
            breakpoint()
            
            eval_loss = 0
            eval_error = []
            
            pred_depths = []
            gt_depths = []
            bbox_mask_depths = []
            
            bbox_eval_error=[]
            
            vis_attn_map = True
            if vis_attn_map:
                
                ca1_names=[]
                ca1_modules=[]
                ca1_maps=[]
                for name, module in model['depth'].module.named_modules():
                    if 'cross_attn_module1' in name and 'c_attn_drop' in name:
                        ca1_names.append(name)
                        ca1_modules.append(module)
                        module.register_forward_hook(lambda m,i,o: ca1_maps.append(o.detach().cpu()) )
            

            breakpoint()
            # validation loop
            for i, inputs in enumerate(tqdm(val_loader)):
                
                total_loss = 0
                losses = {}
                
                # multiframe validation
                if train_cfg.data.dataset=='kitti_depth_multiframe':
                    # breakpoint()
                    add_noise = train_cfg.add_noise
                    if add_noise:
                        img0 = inputs[0]['color']
                        img0_n = img0.squeeze().permute(1,2,0).detach().cpu().numpy() * 255
                        img0_n = img0_n.astype(np.uint8)
                        
                        noise_type = 'motion_blur'
                        noise_degree = 5
                        img0_corrupt = image_corruption(img0_n, noise_type, noise_degree) 
                        
                        img0_t = torch.tensor(img0_corrupt)
                        img0_t = img0_t.permute(2,0,1) / 255
                        inputs[0]['color'] = img0_t.unsqueeze(dim=0)
                        
                        save_image(img0, '../vis_corrupt_img/img0.jpg')
                        save_image(img0_t, '../vis_corrupt_img/img0_corrupt.jpg')
                        
                    for input in inputs:
                        for key, ipt in input.items():
                            if type(ipt) == torch.Tensor:
                                input[key] = ipt.to(device)     # Place current and previous frames on cuda
                    total_loss, _, pred_depth, pred_uncert, pred_depth_mask = loss.compute_loss_multiframe(inputs, model, train_cfg, EVAL)   
                    # total_loss, _, pred_depth, pred_uncert, pred_depth_mask = loss.compute_loss_multiframe_debugmanydepth(inputs, model, train_cfg, EVAL, encoder)   
                                              
                    gt_depth = inputs[0]['depth_gt']
                    inputs_color = inputs[0]['color']
                    inputs_box = inputs[0]['box']
                    inputs_curr_folder = inputs[0]['curr_folder']
                    inputs_curr_frame = inputs[0]['curr_frame']
                    
                    
                    # ca_maps for cross_attn_modules
                    maps1 = ca1_maps[-4:]
                    
                    # rnd_gen = torch.rand(480).argsort()
                    # to_vis_idx = rnd_gen[:30]
                    to_vis_idx = [92,210,222,251,419]
                    discard_ratio = 0.8
                    
                    breakpoint()
                    for vis_idx in to_vis_idx:
                        pho_visualize_cross_attn_map( ca_module_num=1, iter_num=i, 
                                                      curr_frame=inputs[0]['color'], prev_frame=inputs[1]['color'], 
                                                      ca_maps=maps1, discard_ratio=discard_ratio, vis_idx=vis_idx, 
                                                      save_folder_name=train_cfg.model.save_folder_name)
                            
                    
                    
                # singleframe validation
                else:
                    add_noise = train_cfg.add_noise
                    if add_noise:
                        img0 = inputs['color']
                        img0_n = img0.squeeze().permute(1,2,0).detach().cpu().numpy() * 255
                        img0_n = img0_n.astype(np.uint8)
                        
                        noise_type = 'motion_blur'
                        noise_degree = 5
                        img0_corrupt = image_corruption(img0_n, noise_type, noise_degree) 
                        
                        img0_t = torch.tensor(img0_corrupt)
                        img0_t = img0_t.permute(2,0,1) / 255
                        inputs['color'] = img0_t.unsqueeze(dim=0)
                        
                        save_image(img0, '../vis_corrupt_img/sf_img0.jpg')
                        save_image(img0_t, '../vis_corrupt_img/sf_img0_corrupt.jpg')
                       
                    for key, ipt in inputs.items():
                        if type(ipt) == torch.Tensor:
                            inputs[key] = ipt.to(device)

                    total_loss, _, pred_depth, pred_uncert, pred_depth_mask = loss.compute_loss(inputs, model, train_cfg, EVAL)
                    
                    gt_depth = inputs['depth_gt']
                    inputs_color = inputs['color']
                    inputs_box = inputs['box']
                    inputs_curr_folder = inputs['curr_folder']
                    inputs_curr_frame = inputs['curr_frame']
                    
                eval_loss += total_loss
                # pred_depth.squeeze(dim=1)은 tensor 로 (8,H,W) 이고. pred_depths 는 [] 리스트이다.
                # pred_depths.extend( pred_depth )를 해주면 pred_depth 의 8개의 이미지들이 차례로 리스트로 들어가서 리스트 len은 개가 돼
                # 즉 list = [ pred_img1(H,W), pred_img2(H,W), . . . ] 

                # BBOX
                b, c, h, w = gt_depth.shape

                # normalize [0,1] box coordinates to real image shape
                inputs_box[:,:,1] *= w
                inputs_box[:,:,3] *= w 
                inputs_box[:,:,2] *= h 
                inputs_box[:,:,4] *= h

                # show boxes on real-size image
                real_size_img = F.interpolate(inputs_color, size=(h,w), mode='bilinear')     # (b,3,375,1242)
                bbox_drawn_imgs, bbox_masks = draw_boxes(real_size_img, inputs_box)            
                
                bbox_drawn_imgs_t = torch.stack(bbox_drawn_imgs).to(device)
                bbox_masks_t= torch.stack(bbox_masks).to(device)
                masked_img = bbox_drawn_imgs_t * bbox_masks_t
                masked_pred = pred_depth * bbox_masks_t
                masked_gtdepth = gt_depth * bbox_masks_t
                
                save_img_path = f'../vis_bbox/{train_cfg.model.save_folder_name}'
                if not os.path.exists(save_img_path):
                    os.makedirs(save_img_path)
                             
                # save bbox and gtdepth images every 50 iteration
                if i%25 == 0:
                    for j in range( b ):    # batch_size 만큼 loop
                        save_img_name = f'{inputs_curr_folder[j].split("/")[-1]}_{inputs_curr_frame[j]}'
                        save_image(bbox_drawn_imgs_t[j], f'{save_img_path}/{i}_{save_img_name}_bbox_drawn_img.png', normalize=True)
                        save_image(gt_depth[j], f'{save_img_path}/{i}_{save_img_name}_gtdepth.png', normalize=True)
                        save_image(bbox_masks_t[j], f'{save_img_path}/{i}_{save_img_name}_bbox_masks.png', normalize=True)
                        save_image(masked_img[j], f'{save_img_path}/{i}_{save_img_name}_masked_img.png', normalize=True)
                        save_image(pred_depth[j], f'{save_img_path}/{i}_{save_img_name}_pred_depth.png', normalize=True)
                        save_image(masked_pred[j], f'{save_img_path}/{i}_{save_img_name}_masked_pred.png', normalize=True)
                        save_image(masked_gtdepth[j], f'{save_img_path}/{i}_{save_img_name}_masked_gtdepth.png', normalize=True)
                        

                pred_depths.extend(pred_depth.squeeze(1).cpu().numpy()) # pred_depths=[ (375,1242), (374,1242)]
                gt_depths.extend(gt_depth.squeeze(1).cpu().numpy())
                bbox_mask_depths.extend(bbox_masks_t.squeeze(1).cpu().numpy())
                
            # cv2.imwrite('../tmp1_pred_depths.png', pred_depths[0])
            # cv2.imwrite('../tmp1_gt_depths.png', gt_depths[0]) 
            # cv2.imwrite('../tmp1_bbox_mask_depths.png', bbox_mask_depths[0]*255) 
            
            # cv2.imwrite('../tmp2_pred_depths.png', pred_depths[100])
            # cv2.imwrite('../tmp2_gt_depths.png', gt_depths[100]) 
            # cv2.imwrite('../tmp2_bbox_mask_depths.png', bbox_mask_depths[100]*255) 
            
            # cv2.imwrite('../tmp3_pred_depths.png', pred_depths[300])
            # cv2.imwrite('../tmp3_gt_depths.png', gt_depths[300]) 
            # cv2.imwrite('../tmp3_bbox_mask_depths.png', bbox_mask_depths[300]*255) 
            
            eval_error = eval_metric(pred_depths, gt_depths, train_cfg)  
            error_dict = get_eval_dict(eval_error)
            # error_dict["val_loss"] = eval_loss / len(val_loader)  
            print(error_dict)

            bbox_eval_error = eval_metric_bbox(pred_depths, gt_depths, train_cfg, bbox_mask_depths)
            bbox_error_dict = get_eval_dict(bbox_eval_error)
            print(bbox_error_dict)

            with open(f'{save_img_path}/result.txt', 'w') as f:
                f.write('------------------full eval metric------------------\n')
                for key, val in error_dict.items():
                    f.write(f'{key}: {np.round(val,5)} \n')
                
                f.write('\n')
                f.write('------------------bbox eval metric------------------\n')
                for key, val in bbox_error_dict.items():
                    f.write(f'{key}: {np.round(val,5)} \n')
            
            breakpoint()
            print('FINISHED')
            
            # if train_cfg.data.dataset=='kitti_depth_multiframe':
            #     visualize(inputs[0], pred_depth, pred_depth_mask, pred_uncert, wandb) 
            # else:
            #     visualize(inputs, pred_depth, pred_depth_mask, pred_uncert, wandb)  
            