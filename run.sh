export PYTHONPATH=/home/zhfeing/project/cv-lib-PyTorch
export TORCH_HOME=/home/zhfeing/model-zoo

seed=1029

port=9201
export CUDA_VISIBLE_DEVICES=6

python dist_engine.py \
    --num-nodes 1 \
    --rank 0 \
    --master-url tcp://localhost:${port} \
    --backend nccl \
    --seed ${seed} \
    --multiprocessing \
    --file-name-cfg loss_landscape \
    --cfg-filepath config/val/plot-cnn.yaml \
    --log-dir run/cifar100/val/agent \
    --worker plot_2D_worker &
