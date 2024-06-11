conda create -n mfdepth python=3.9
conda activate mfdepth

# CUDA 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install tqdm
pip install wandb
pip install scikit-image
pip install opencv-python
pip install einops
pip install timm