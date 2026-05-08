from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BitNetReplacementReport:
    replaced_modules: int
    replaced_parameters: int
    skipped_modules: int


def _fake_quantize_with_ste(
    tensor: torch.Tensor,
    quantized_tensor: torch.Tensor,
) -> torch.Tensor:
    return tensor + (quantized_tensor.to(dtype=tensor.dtype) - tensor).detach()


def _absmean_weight_quantize(
    weight: torch.Tensor,
    bits: float,
    eps: float,
) -> torch.Tensor:
    scale = weight.detach().float().abs().mean().clamp_min(eps)
    weight_fp32 = weight.float()
    if bits == 1:
        quantized = weight_fp32.sign()
        quantized = torch.where(quantized == 0, torch.ones_like(quantized), quantized)
    elif bits == 1.58:
        quantized = torch.round(weight_fp32 / scale).clamp(-1, 1)
    else:
        raise ValueError("BitNet weight_bits must be 1 or 1.58.")
    return _fake_quantize_with_ste(weight, quantized * scale)


def _absmax_activation_quantize(
    activation: torch.Tensor,
    bits: int,
    eps: float,
) -> torch.Tensor:
    if not activation.is_floating_point():
        return activation
    qmax = (2 ** (bits - 1)) - 1
    if qmax < 1:
        raise ValueError("BitNet activation_bits must be >= 2.")
    activation_fp32 = activation.float()
    scale = activation_fp32.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = scale / qmax
    quantized = torch.round(activation_fp32 / scale).clamp(-qmax, qmax) * scale
    return _fake_quantize_with_ste(activation, quantized)


class BitLinear(nn.Module):
    """BitNet-style fake-quantized linear layer for quantization-aware training."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: float = 1.58,
        activation_bits: int = 8,
        quantize_activations: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if weight_bits not in {1, 1.0, 1.58}:
            raise ValueError("BitLinear supports weight_bits of 1 or 1.58.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = 1 if weight_bits in {1, 1.0} else 1.58
        self.activation_bits = activation_bits
        self.quantize_activations = quantize_activations
        self.eps = eps
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        weight_bits: float,
        activation_bits: int,
        quantize_activations: bool,
        eps: float,
    ) -> BitLinear:
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            quantize_activations=quantize_activations,
            eps=eps,
        )
        module.weight = nn.Parameter(linear.weight.detach().clone())
        module.weight.requires_grad = linear.weight.requires_grad
        if linear.bias is not None and module.bias is not None:
            module.bias = nn.Parameter(linear.bias.detach().clone())
            module.bias.requires_grad = linear.bias.requires_grad
        return module.to(device=linear.weight.device, dtype=linear.weight.dtype)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = fan_in**-0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        weight = _absmean_weight_quantize(self.weight, self.weight_bits, self.eps)
        if self.quantize_activations:
            input_tensor = _absmax_activation_quantize(
                input_tensor,
                self.activation_bits,
                self.eps,
            )
        return F.linear(input_tensor, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_bits={self.weight_bits}, "
            f"activation_bits={self.activation_bits}, "
            f"quantize_activations={self.quantize_activations}"
        )


def _set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, nn.Sequential | nn.ModuleList):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def apply_bitnet_quantization(
    model: nn.Module,
    config: Any,
) -> BitNetReplacementReport:
    weight_bits = float(config.get("weight_bits", 1.58))
    activation_bits = int(config.get("activation_bits", 8))
    quantize_activations = bool(config.get("quantize_activations", True))
    only_trainable = bool(config.get("only_trainable", True))
    min_features = int(config.get("min_features", 0))
    eps = float(config.get("eps", 1e-5))
    exclude_name_contains = tuple(config.get("exclude_name_contains", []))

    replacements: list[tuple[str, nn.Linear]] = []
    skipped_modules = 0
    for name, module in model.named_modules():
        if name == "" or isinstance(module, BitLinear) or not isinstance(module, nn.Linear):
            continue
        if any(excluded in name for excluded in exclude_name_contains):
            skipped_modules += 1
            continue
        if only_trainable and not any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        ):
            skipped_modules += 1
            continue
        if min(module.in_features, module.out_features) < min_features:
            skipped_modules += 1
            continue
        replacements.append((name, module))

    replaced_parameters = 0
    for name, module in replacements:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        bitlinear = BitLinear.from_linear(
            module,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
            quantize_activations=quantize_activations,
            eps=eps,
        )
        replaced_parameters += sum(parameter.numel() for parameter in module.parameters())
        _set_child_module(parent, child_name, bitlinear)

    return BitNetReplacementReport(
        replaced_modules=len(replacements),
        replaced_parameters=replaced_parameters,
        skipped_modules=skipped_modules,
    )
