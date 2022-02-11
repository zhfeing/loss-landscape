export PYTHONPATH=/home/zhfeing/project/cv-lib-PyTorch
export TORCH_HOME=/home/zhfeing/model-zoo

seed=1029

port=9902
export CUDA_VISIBLE_DEVICES=0

python convert_mutual.py \
    --mutual-ckpt ~/nfs3/joint-vit-s-cifar100-best.pth \
    --out-dir run/unpack_joint-vit-s-cifar100
