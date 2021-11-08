import os
import logging
import shutil
import time
from collections import OrderedDict
from logging.handlers import QueueHandler
from typing import Dict, Any, Iterable, List, Tuple
import datetime
import yaml

import torch
from torch import nn, Tensor
import torch.cuda
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torch.cuda import amp

import cv_lib.utils as utils
import cv_lib.distributed.utils as dist_utils

from vit_mutual.data import build_train_dataset
from vit_mutual.models import get_model
from vit_mutual.eval import Evaluation
import vit_mutual.utils as vit_utils

from loss_landscape.utils import DistLaunchArgs, LogArgs
from loss_landscape.direction import setup_direction
from loss_landscape.surface import setup_coordinates


def plot_2D_worker(
    gpu_id: int,
    launch_args: DistLaunchArgs,
    log_args: LogArgs,
    global_cfg: Dict[str, Any]
):
    """
    What created in this function is only used in this process and not shareable
    """
    # setup process root logger
    if launch_args.distributed:
        root_logger = logging.getLogger()
        handler = QueueHandler(log_args.logger_queue)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False

    # split configs
    data_cfg: Dict[str, Any] = global_cfg["dataset"]
    train_cfg: Dict[str, Any] = global_cfg["training"]
    val_cfg: Dict[str, Any] = global_cfg["validation"]
    model_cfg: Dict[str, Any] = global_cfg["model"]
    plot_cfg: Dict[str, Any] = global_cfg["plot"]

    distributed = launch_args.distributed
    # get current rank
    current_rank = launch_args.rank
    if distributed:
        if launch_args.multiprocessing:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            current_rank = launch_args.rank * launch_args.ngpus_per_node + gpu_id
        dist.init_process_group(
            backend=launch_args.backend,
            init_method=launch_args.master_url,
            world_size=launch_args.world_size,
            rank=current_rank
        )

    assert dist_utils.get_rank() == current_rank, "code bug"
    # set up process logger
    logger = logging.getLogger("worker_rank_{}".format(current_rank))

    if current_rank == 0:
        logger.info("Starting with configs:\n%s", yaml.dump(global_cfg))

    # make determinstic
    if launch_args.seed is not None:
        seed = launch_args.seed + current_rank
        logger.info("Initial rank %d with seed: %d", current_rank, seed)
        utils.make_deterministic(seed)
    # set cuda
    torch.backends.cudnn.benchmark = True
    logger.info("Use GPU: %d for training", gpu_id)
    device = torch.device("cuda:{}".format(gpu_id))
    # IMPORTANT! for distributed training (reduce_all_object)
    torch.cuda.set_device(device)

    # get dataloader
    logger.info("Building dataset...")
    train_loader, val_loader, n_classes = build_train_dataset(
        data_cfg,
        train_cfg,
        val_cfg,
        launch_args
    )
    # create model
    logger.info("Building model...")
    model = get_model(model_cfg, n_classes)
    logger.info(
        "Built model with %d parameters, %d trainable parameters",
        utils.count_parameters(model, include_no_grad=True),
        utils.count_parameters(model, include_no_grad=False)
    )
    if train_cfg.get("pre_train", None) is not None:
        vit_utils.load_pretrain_model(
            pretrain_fp=train_cfg["pre_train"],
            model=model,
        )
        logger.info("Loaded pretrain model: %s", train_cfg["pre_train"])
    model.to(device)
    # model_without_ddp = model
    if distributed:
        if train_cfg.get("sync_bn", False):
            logger.warning("Convert model `BatchNorm` to `SyncBatchNorm`")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])

    # setup direction
    direction_fp = os.path.join(log_args.logdir, "direction.pth")
    if dist_utils.is_main_process():
        setup_direction(
            direction_fp=direction_fp,
            model=model,
            override=launch_args.override,
            **plot_cfg["direction"]
        )
    # setup surface
    surface_file = os.path.join(log_args.logdir, "surface.pth")
    if dist_utils.is_main_process():
        setup_coordinates(
            surface_fp=surface_file,
            direction_fp=direction_fp,
            override=launch_args.override,
            coordinate_cfg=plot_cfg["coordinate"],
            device=device
        )
    dist_utils.barrier()
    directions = torch.load(direction_fp, map_location="cuda")

