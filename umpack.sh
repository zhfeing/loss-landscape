export PYTHONPATH=/home/zhfeing/project/cv-lib-PyTorch
export TORCH_HOME=/home/zhfeing/model-zoo

seed=1029

port=9902
export CUDA_VISIBLE_DEVICES=0

python convert_mutual.py \
    --mutual-ckpt ~/project/ViT-research/run/joint-demo-s-1029/ckpt/best.pth \
    --out-dir run/unpack_joint_base
