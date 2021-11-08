from collections import OrderedDict
import logging
import os
from typing import Dict, Any

import torch


def setup_coordinates(
    surface_fp: str,
    direction_fp: str,
    coordinate_cfg: Dict[str, Any],
    device: torch.device,
    override: bool = True
):
    logger = logging.getLogger("setup_direction")
    logger.info("Setting up directions")

    if os.path.isfile(surface_fp) and not override:
        logger.info("Surface file already exists")
        return

    coordinates = OrderedDict()
    x_coordinates = torch.linspace(device=device, **coordinate_cfg["x_axis"])
    coordinates["x_coordinates"] = x_coordinates
    y_coordinates = torch.linspace(device=device, **coordinate_cfg["y_axis"])
    coordinates["y_coordinates"] = y_coordinates
    torch.save(coordinates, direction_fp)
    logger.info("Write coordinates done")
