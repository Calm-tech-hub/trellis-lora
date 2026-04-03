from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f'LoRA rank must be positive, got {rank}')

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
        if base_layer.bias is not None:
            self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_down = nn.Linear(self.in_features, rank, bias=False, device=base_layer.weight.device, dtype=torch.float32)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False, device=base_layer.weight.device, dtype=torch.float32)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        x_lora = x.to(self.lora_down.weight.dtype)
        update = self.lora_down(x_lora)
        update = self.dropout(update)
        update = self.lora_up(update).to(base.dtype) * self.scale
        return base + update


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def apply_lora_to_linear_layers(
    model: nn.Module,
    target_keywords: Iterable[str],
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> list[str]:
    replaced = []
    target_keywords = tuple(target_keywords)

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(keyword in module_name for keyword in target_keywords):
            continue
        parent, child_name = _get_parent_module(model, module_name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(module_name)

    return replaced
