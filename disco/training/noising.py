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

from disco.data.ccd import MASK_REF_CHARGE, MASK_REF_MASK, MASK_REF_POS
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


def _as_batched(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0), True
    if tensor.ndim == 2 and tensor.shape[-1] == 3:
        return tensor.unsqueeze(0), True
    return tensor, False


def _restore_batch(tensor: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return tensor.squeeze(0) if squeezed else tensor


def remask_masked_reference_features(
    feature_dict: dict[str, torch.Tensor],
    sequence_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Masks ref_pos/ref_charge/ref_mask for stochastically masked residues.

    ``sequence_mask`` is indexed over protein residues only. The feature dict
    carries token-level ``prot_residue_mask`` and atom-level ``atom_to_token_idx``
    / ``atom_to_tokatom_idx``; together these let us map masked protein residues
    back to their backbone atoms and replace the reference features with the
    canonical masked-residue references used by ``TaskManager``.
    """
    required = [
        "ref_pos",
        "ref_charge",
        "ref_mask",
        "prot_residue_mask",
        "atom_to_token_idx",
        "atom_to_tokatom_idx",
    ]
    missing = [key for key in required if key not in feature_dict]
    if missing:
        raise KeyError(
            "Cannot remask reference features; missing feature(s): "
            + ", ".join(missing)
        )

    ref_pos, squeeze_ref_pos = _as_batched(feature_dict["ref_pos"])
    ref_charge, squeeze_ref_charge = _as_batched(feature_dict["ref_charge"])
    ref_mask, squeeze_ref_mask = _as_batched(feature_dict["ref_mask"])
    prot_residue_mask, _ = _as_batched(feature_dict["prot_residue_mask"])
    atom_to_token_idx, _ = _as_batched(feature_dict["atom_to_token_idx"])
    atom_to_tokatom_idx, _ = _as_batched(feature_dict["atom_to_tokatom_idx"])

    if sequence_mask.ndim == 1:
        sequence_mask = sequence_mask.unsqueeze(0)
    sequence_mask = sequence_mask.to(device=ref_pos.device, dtype=torch.bool)

    batch_size, n_token = prot_residue_mask.shape
    full_token_mask = torch.zeros(
        (batch_size, n_token),
        device=ref_pos.device,
        dtype=torch.bool,
    )
    for batch_idx in range(batch_size):
        prot_token_idx = torch.nonzero(
            prot_residue_mask[batch_idx].to(dtype=torch.bool),
            as_tuple=False,
        ).squeeze(-1)
        if sequence_mask.shape[-1] < prot_token_idx.shape[-1]:
            raise ValueError(
                "sequence_mask length must cover the number of protein tokens: "
                f"{sequence_mask.shape[-1]} < {prot_token_idx.shape[-1]}"
            )
        full_token_mask[batch_idx, prot_token_idx] = sequence_mask[
            batch_idx,
            : prot_token_idx.shape[-1],
        ]

    batch_indices = torch.arange(batch_size, device=ref_pos.device)[:, None]
    atom_token_mask = full_token_mask[batch_indices, atom_to_token_idx.long()]
    backbone_atom_mask = (0 <= atom_to_tokatom_idx) & (atom_to_tokatom_idx < 4)
    atom_mask = atom_token_mask & backbone_atom_mask.to(dtype=torch.bool)

    mask_ref_pos = torch.as_tensor(
        MASK_REF_POS,
        device=ref_pos.device,
        dtype=ref_pos.dtype,
    )
    mask_ref_charge = torch.as_tensor(
        MASK_REF_CHARGE,
        device=ref_charge.device,
        dtype=ref_charge.dtype,
    )
    mask_ref_mask = torch.as_tensor(
        MASK_REF_MASK,
        device=ref_mask.device,
        dtype=ref_mask.dtype,
    )

    replacement_idx = atom_to_tokatom_idx.long().clamp(min=0, max=3)
    ref_pos = ref_pos.clone()
    ref_charge = ref_charge.clone()
    ref_mask = ref_mask.clone()
    ref_pos[atom_mask] = mask_ref_pos[replacement_idx[atom_mask]]
    ref_charge[atom_mask] = mask_ref_charge[replacement_idx[atom_mask]]
    ref_mask[atom_mask] = mask_ref_mask[replacement_idx[atom_mask]]

    feature_dict = dict(feature_dict)
    feature_dict["ref_pos"] = _restore_batch(ref_pos, squeeze_ref_pos)
    feature_dict["ref_charge"] = _restore_batch(ref_charge, squeeze_ref_charge)
    feature_dict["ref_mask"] = _restore_batch(ref_mask, squeeze_ref_mask)
    return feature_dict
