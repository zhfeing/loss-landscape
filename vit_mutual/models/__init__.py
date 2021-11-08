from typing import Any, Dict

import torch.nn as nn

from cv_lib.classification.models import get_model as get_cnn_official_models
from .vision_transformers import get_vit, get_deit
from .cnn import get_cnn


__REGISTERED_MODELS__ = {
    "vit": get_vit,
    "deit": get_deit,
    "cnn": get_cnn,
    "official_models": get_cnn_official_models
}


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        if isinstance(output, dict):
            return output
        ret = {
            "pred": output
        }
        return ret


def get_model(model_cfg: Dict[str, Any], num_classes: int, with_wrapper: bool = True) -> nn.Module:
    model = __REGISTERED_MODELS__[model_cfg["name"]](model_cfg, num_classes)
    if with_wrapper:
        model = ModelWrapper(model)
    return model
