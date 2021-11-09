import logging
from typing import Dict, Any

import torch.utils.data as data

import cv_lib.classification.data as cls_data
import cv_lib.distributed.utils as dist_utils

from .aug import get_data_aug


def build_dataset(
    data_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
):
    logger = logging.getLogger("build_dataset")
    # get dataloader
    train_aug = get_data_aug(data_cfg["name"], "train")
    train_dataset, _, n_classes = cls_data.get_dataset(
        data_cfg,
        train_aug,
        None
    )
    if dist_utils.is_main_process():
        logger.info(
            "Loaded %s dataset with %d train examples, %d classes",
            data_cfg["name"], len(train_dataset), n_classes
        )
    train_sampler = data.SequentialSampler(train_dataset)
    train_bs = train_cfg["batch_size"]
    train_workers = train_cfg["num_workers"]
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        num_workers=train_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    logger.info(
        "Build train dataset done\nTraining: %d imgs, %d batchs",
        len(train_dataset),
        len(train_loader),
    )
    return train_loader, n_classes

