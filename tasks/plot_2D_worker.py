import os
import logging
import time
from logging.handlers import QueueHandler
from typing import Dict, Any
import datetime
import yaml

import torch
import torch.nn as nn
import torch.cuda
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.backends.cudnn
from torch.cuda import amp

import cv_lib.utils as utils
import cv_lib.distributed.utils as dist_utils

from vit_mutual.models import get_model
from vit_mutual.loss import get_loss_fn
from vit_mutual.eval import Evaluation
import vit_mutual.utils as vit_utils

from loss_landscape.utils import DistLaunchArgs, LogArgs
from loss_landscape.direction import setup_direction
from loss_landscape.surface import setup_coordinates
from data import build_dataset


class PlotWorker:
    def __init__(
        self,
        coordinate_fp: str,
        data_loader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        evaluator: Evaluation,
        device: torch.device
    ):
        coordinates = torch.load(coordinate_fp, map_location=device)
        self.x_coordinates = coordinates["x_coordinates"]
        self.y_coordinates = coordinates["y_coordinates"]
        self.run_groups = len(self.x_coordinates), len(self.y_coordinates)

        self.model = model
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.evaluator = evaluator

        self.loss_map = torch.empty(size=self.run_groups, device=device)
        self.acc_map = torch.empty_like(self.loss_map)
        self.device = device

        self._assign_tasks()

    def _assign_tasks(self):
        x_steps = torch.arange(self.run_groups[0])
        y_steps = torch.arange(self.run_groups[1])
        assignment_ids = torch.cartesian_prod(x_steps, y_steps)
        assignments = torch.arange(0, assignment_ids.shape[0]) % dist_utils.get_world_size() == dist_utils.get_rank()

    def __call__(self):
        self.evaluator(self.model)


def plot_2D_worker(
    gpu_id: int,
    launch_args: DistLaunchArgs,
    log_args: LogArgs,
    global_cfg: Dict[str, Any]
):
    ################################################################################
    # Initialization
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
    model_cfg: Dict[str, Any] = global_cfg["model"]
    loss_cfg: Dict[str, Any] = global_cfg["loss"]
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
    train_loader, n_classes = build_dataset(data_cfg, train_cfg)
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
    if distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id])
    loss_fn = get_loss_fn(loss_cfg).to(device)
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
    coordinates_fp = os.path.join(log_args.logdir, "coordinates.pth")
    if dist_utils.is_main_process():
        setup_coordinates(
            coordinates_fp=coordinates_fp,
            override=launch_args.override,
            coordinate_cfg=plot_cfg["coordinate"],
            device=device
        )
    evaluator = Evaluation(
        loss_fn=loss_fn,
        val_loader=train_loader,
        loss_weights=loss_cfg["weight_dict"],
        device=device
    )
    worker = PlotWorker(
        coordinate_fp=coordinates_fp,
        data_loader=train_loader,
        model=model,
        loss_fn=loss_fn,
        evaluator=evaluator,
        device=device
    )
    worker()
