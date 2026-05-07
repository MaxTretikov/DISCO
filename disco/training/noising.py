# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import dataclass

import torch

from disco.data.constants import MASK_TOKEN_IDX


@dataclass
class NoisedBatch:
    """Training-time noised sequence and structure state."""

    xt_struct: torch.Tensor
    xt_seq: torch.Tensor
    structure_noise: torch.Tensor
    sequence_time: torch.Tensor
    sequence_conditioning: torch.Tensor
    sequence_mask: torch.Tensor


def sample_structure_noise(
    batch_size: int,
    device: torch.device,
    sigma_data: float = 16.0,
    mean: float = -1.2,
    std: float = 1.5,
) -> torch.Tensor:
    """Samples the paper's log-normal structure noise level."""
    return sigma_data * torch.exp(mean + std * torch.randn(batch_size, device=device))


def sample_sequence_time(
    batch_size: int,
    device: torch.device,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Samples masked-diffusion sequence time r in [eps, 1]."""
    return torch.rand(batch_size, device=device).clamp_min(eps)


def apply_structure_noise(
    x0_struct: torch.Tensor,
    sigma: torch.Tensor,
    coord_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Adds Gaussian coordinate noise, optionally preserving invalid coordinates."""
    noise = sigma[..., None, None] * torch.randn_like(x0_struct)
    if coord_mask is not None:
        noise = noise * coord_mask.to(dtype=x0_struct.dtype).unsqueeze(-1)
    return x0_struct + noise


def apply_sequence_mask(
    true_seq: torch.Tensor,
    sequence_time: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks protein tokens independently with probability r."""
    if true_seq.ndim == 1:
        true_seq = true_seq.unsqueeze(0)

    if sequence_time.ndim == 0:
        sequence_time = sequence_time.unsqueeze(0)

    mask = torch.rand(true_seq.shape, device=true_seq.device) < sequence_time[:, None]
    if valid_mask is not None:
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        mask = mask & valid_mask.to(device=true_seq.device, dtype=torch.bool)

    xt_seq = true_seq.masked_fill(mask, MASK_TOKEN_IDX)
    return xt_seq, mask


def sequence_time_to_conditioning(
    sequence_time: torch.Tensor,
    model,
    mode: str = "scheduler",
) -> torch.Tensor:
    """Maps raw sequence time r to the scalar consumed by DiffusionConditioning.

    The paper describes sampling raw r, but the released inference path feeds an
    EDM-like noise level to the diffusion module. This function keeps that choice
    explicit so it can be ablated.
    """
    if mode == "raw":
        return sequence_time
    if mode == "sigma_data_scaled":
        return sequence_time * model.configs.sigma_data
    if mode == "scheduler":
        return model.inference_noise_scheduler.time_to_noise_lvl(sequence_time)
    raise ValueError(f"Unknown sequence conditioning mode: {mode}")


def make_noised_batch(
    x0_struct: torch.Tensor,
    true_seq: torch.Tensor,
    model,
    coord_mask: torch.Tensor | None = None,
    valid_seq_mask: torch.Tensor | None = None,
    sequence_conditioning_mode: str = "scheduler",
) -> NoisedBatch:
    """Builds the multimodal noised state used by one training step."""
    batch_size = x0_struct.shape[0] if x0_struct.ndim == 3 else 1
    device = x0_struct.device

    structure_noise = sample_structure_noise(
        batch_size=batch_size,
        device=device,
        sigma_data=model.configs.sigma_data,
    )
    sequence_time = sample_sequence_time(batch_size=batch_size, device=device)

    if x0_struct.ndim == 2:
        x0_struct = x0_struct.unsqueeze(0)
    xt_struct = apply_structure_noise(x0_struct, structure_noise, coord_mask)
    xt_seq, sequence_mask = apply_sequence_mask(true_seq, sequence_time, valid_seq_mask)
    sequence_conditioning = sequence_time_to_conditioning(
        sequence_time,
        model,
        mode=sequence_conditioning_mode,
    )

    return NoisedBatch(
        xt_struct=xt_struct,
        xt_seq=xt_seq,
        structure_noise=structure_noise,
        sequence_time=sequence_time,
        sequence_conditioning=sequence_conditioning,
        sequence_mask=sequence_mask,
    )

