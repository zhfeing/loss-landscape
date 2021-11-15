export PYTHONPATH=/home/zhfeing/project/cv-lib-PyTorch
export TORCH_HOME=/home/zhfeing/model-zoo

seed=1029

port=9902
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7

python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:${port} \
    --backend nccl \
    --seed ${seed} \
    --multiprocessing \
    --file-name-cfg loss_landscape \
    --cfg-filepath config/plot-vit_mutual_base.yaml \
    --log-dir run/cifar100/vit-mutual-base \
    --worker plot_2D_worker &
