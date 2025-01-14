from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn

import vit_mutual.models.layers as layers


############################################################################################
# MLP Block

class MLP_CNN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dim_feedforward: int,
        activation: str,
        dropout: float = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv2d(dim_feedforward, embed_dim, kernel_size=1)
        self.dropout = layers.get_dropout(dropout)
        self.activation = layers.get_activation_fn(activation)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.normal_(self.conv1.bias, 1e-6)
        nn.init.normal_(self.conv2.bias, 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.conv2(x)
        return x


class MLP_BottleNeck(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        dim_feedforward: int,
        activation: str,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, dim_feedforward, kernel_size=1)
        self.conv2 = nn.Conv2d(dim_feedforward, dim_feedforward, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(dim_feedforward, embed_dim, kernel_size=1)
        self.activation1 = layers.get_activation_fn(activation)
        self.activation2 = layers.get_activation_fn(activation)
        self.norm1 = nn.BatchNorm2d(dim_feedforward)
        self.norm2 = nn.BatchNorm2d(dim_feedforward)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.conv3.bias is not None:
            nn.init.zeros_(self.conv3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation1(self.norm1(self.conv1(x)))
        x = self.activation2(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        return x


class DummyMLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


__REGISTERED_MLP__ = {
    "mlp_cnn": MLP_CNN,
    "mlp_bottleneck": MLP_BottleNeck,
    "dummy_mlp": DummyMLP,
}


def get_mlp(
    embed_dim: int,
    activation: str,
    mlp_cfg: Dict[str, Any]
):
    mlp_cfg = deepcopy(mlp_cfg)
    name = mlp_cfg.pop("name")
    mlp = __REGISTERED_MLP__[name](
        embed_dim=embed_dim,
        activation=activation,
        **mlp_cfg
    )
    return mlp

