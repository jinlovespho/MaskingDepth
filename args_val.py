import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description='args')

    # Training args 
    parser.add_argument('--num_epoch',      type=int,   default=50)
    parser.add_argument('--batch_size',     type=int,   default=8)
    parser.add_argument('--learning_rate',  type=float, default=1e-4)
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--seed',           type=int,   default=42)


    # Depth args 
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=80.0)


    # Model args 
    parser.add_argument('--vit_type', type=str, default='vit_base')
    parser.add_argument('--masking_ratio', type=float)


    # Save args 
    parser.add_argument("--epoch_save_freq", type=int, default=5)


    # Data args 
    parser.add_argument('--data_path',      type=str,   default='./path/to/dataset')
    parser.add_argument("--dataset",        type=str,   choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"], default='kitti')
    parser.add_argument("--split",          type=str,   choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],  default="eigen_zhou")
    parser.add_argument('--img_ext',        type=str,   default='.jpg')
    parser.add_argument('--resized_height', type=int,   default=192)
    parser.add_argument('--resized_width',  type=int,   default=640)


    # Logging args 
    parser.add_argument('--wandb',          action='store_true' )
    parser.add_argument('--wdb_proj_name',  type=str)
    parser.add_argument('--wdb_exp_name',   type=str)
    parser.add_argument('--log_path',       type=str, default='./path/to/log')


    # Etc args 
    parser.add_argument("--num_layers",
                                type=int,
                                help="number of resnet layers",
                                default=18,
                                choices=[18, 34, 50, 101, 152])

    parser.add_argument("--disparity_smoothness",
                                type=float,
                                help="disparity smoothness weight",
                                default=1e-3)
    parser.add_argument("--scales",
                                nargs="+",
                                type=int,
                                help="scales used in the loss",
                                default=[0, 1, 2, 3])

    parser.add_argument("--use_stereo",
                                help="if set, uses stereo pair for training",
                                action="store_true")
    parser.add_argument("--frame_ids",
                                nargs="+",
                                type=int,
                                help="frames to load",
                                default=[0, -1, 1])


    parser.add_argument("--conf", type=str, help="configuration file path", default="./conf/base_train.yaml")



    args = parser.parse_args()

    return args