import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description='args')

    # Data args 
    parser.add_argument('--data_path',      type=str,   default='/path/to/data')
    parser.add_argument("--dataset",        type=str,   choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", 'kitti_depth_multiframe'])
    parser.add_argument("--splits",         type=str,   choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "eigen_temp"])
    parser.add_argument('--img_ext',        type=str)
    parser.add_argument('--re_height',      type=int,   default=192)
    parser.add_argument('--re_width',       type=int,   default=640)
    
    # Training args 
    parser.add_argument('--num_epoch',      type=int)  
    parser.add_argument('--batch_size',     type=int)
    parser.add_argument('--learning_rate',  type=float) 
    parser.add_argument('--num_workers',    type=int) 
    parser.add_argument('--seed',           type=int)
    
    # Depth args 
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=80.0)

    # Loss args
    parser.add_argument('--training_loss', type=str)
    parser.add_argument('--use_future_frame', action='store_true')

    # Model args 
    parser.add_argument('--model_info', type=str)
    parser.add_argument('--vit_type', type=str, default='vit_base')
    parser.add_argument('--pretrained_weight', type=str)
    parser.add_argument('--num_prev_frame', type=int)
    parser.add_argument('--cross_attn_depth', type=int)
    parser.add_argument('--masking_ratio', type=float)


    # Save args 
    parser.add_argument("--epoch_save_freq", type=int, default=5)


    # Logging args 
    parser.add_argument('--log_tool',         type=str)
    parser.add_argument('--wandb_proj_name',  type=str)
    parser.add_argument('--wandb_exp_name',   type=str)
    parser.add_argument('--log_path',         type=str,     default='./path/to/log')
    
    args = parser.parse_args()

    return args