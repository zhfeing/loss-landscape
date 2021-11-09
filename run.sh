export PYTHONPATH=/home/zhfeing/project/cv-lib-PyTorch
export TORCH_HOME=/home/zhfeing/model-zoo

seed=1029
port=9901
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:${port} \
    --backend nccl \
    --seed ${seed} \
    --multiprocessing \
    --file-name-cfg loss_landscape \
    --cfg-filepath config/plot_2D.yml \
    --log-dir run/cifar100/efficientnet \
    --worker plot_2D_worker
