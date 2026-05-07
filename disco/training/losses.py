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
import torch.nn.functional as F
from openfold.utils.loss import distogram_loss as openfold_distogram_loss


@dataclass
class LossBreakdown:
    """Individual training losses and the weighted total."""

    total: torch.Tensor
    sequence: torch.Tensor
    mse: torch.Tensor
    smooth_lddt: torch.Tensor
    distogram: torch.Tensor

    def as_log_dict(self) -> dict[str, torch.Tensor]:
        return {
            "loss/total": self.total.detach(),
            "loss/sequence": self.sequence.detach(),
            "loss/mse": self.mse.detach(),
            "loss/smooth_lddt": self.smooth_lddt.detach(),
            "loss/distogram": self.distogram.detach(),
        }


def _ensure_batched_coords(coords: torch.Tensor) -> torch.Tensor:
    return coords.unsqueeze(0) if coords.ndim == 2 else coords


def _ensure_batched_mask(mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    if mask.ndim == 1:
        mask = mask.unsqueeze(0).expand(batch_size, -1)
    return mask


def weighted_rigid_align(
    target: torch.Tensor,
    pred: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Aligns target coordinates onto predicted coordinates with weighted Kabsch."""
    target = _ensure_batched_coords(target)
    pred = _ensure_batched_coords(pred)
    weights = _ensure_batched_mask(weights, target.shape[0]).to(dtype=target.dtype)

    weights_sum = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    normalized = weights / weights_sum

    target_center = (target * normalized[..., None]).sum(dim=-2, keepdim=True)
    pred_center = (pred * normalized[..., None]).sum(dim=-2, keepdim=True)
    target_centered = target - target_center
    pred_centered = pred - pred_center

    covariance = torch.matmul(
        (target_centered * weights[..., None]).transpose(-1, -2),
        pred_centered,
    )
    u, _, vh = torch.linalg.svd(covariance.float())
    rotation = torch.matmul(vh.transpose(-1, -2), u.transpose(-1, -2))

    det = torch.linalg.det(rotation)
    correction = torch.ones((*rotation.shape[:-2], 3), device=rotation.device)
    correction[..., -1] = torch.where(det < 0, -1.0, 1.0)
    rotation = torch.matmul(
        vh.transpose(-1, -2) * correction[..., None, :],
        u.transpose(-1, -2),
    ).to(dtype=target.dtype)

    return torch.matmul(target_centered, rotation) + pred_center


def atom_weights_from_features(
    feature_dict: dict[str, torch.Tensor],
    coord_mask: torch.Tensor,
    dna_weight: float = 5.0,
    rna_weight: float = 5.0,
    ligand_weight: float = 10.0,
) -> torch.Tensor:
    """Builds the paper's atom-level structure loss weights."""
    weights = torch.ones_like(coord_mask, dtype=torch.float32)
    for key, extra_weight in [
        ("is_dna", dna_weight),
        ("is_rna", rna_weight),
        ("is_ligand", ligand_weight),
    ]:
        if key in feature_dict:
            mask = feature_dict[key].to(device=coord_mask.device, dtype=torch.bool)
            if mask.ndim == 1 and coord_mask.ndim == 2:
                mask = mask.unsqueeze(0).expand_as(coord_mask)
            weights = weights + mask.to(dtype=weights.dtype) * extra_weight
    return weights * coord_mask.to(dtype=weights.dtype)


def weighted_aligned_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    atom_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Paper weighted aligned MSE, excluding unresolved atoms."""
    pred = _ensure_batched_coords(pred)
    target = _ensure_batched_coords(target)
    coord_mask = _ensure_batched_mask(coord_mask, pred.shape[0]).to(dtype=pred.dtype)
    atom_weights = _ensure_batched_mask(atom_weights, pred.shape[0]).to(dtype=pred.dtype)

    align_weights = atom_weights * coord_mask
    aligned_target = weighted_rigid_align(target, pred.detach(), align_weights, eps=eps)
    per_atom = ((pred - aligned_target) ** 2).sum(dim=-1) / 3.0
    weighted = per_atom * atom_weights * coord_mask
    denom = (atom_weights * coord_mask).sum(dim=-1).clamp_min(eps)
    return (weighted.sum(dim=-1) / denom).mean()


def smooth_lddt_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    coord_mask: torch.Tensor,
    is_dna: torch.Tensor | None = None,
    is_rna: torch.Tensor | None = None,
    default_radius: float = 15.0,
    nucleotide_radius: float = 30.0,
    temperature: float = 0.25,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable LDDT-style coordinate loss from the paper description."""
    pred = _ensure_batched_coords(pred)
    target = _ensure_batched_coords(target)
    batch_size = pred.shape[0]
    coord_mask = _ensure_batched_mask(coord_mask, batch_size).to(dtype=torch.bool)

    pred_d = torch.cdist(pred.float(), pred.float())
    target_d = torch.cdist(target.float(), target.float())
    dist_error = (pred_d - target_d).abs()

    thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0], device=pred.device)
    score = torch.sigmoid((thresholds - dist_error[..., None]) / temperature).mean(-1)

    pair_mask = coord_mask[..., :, None] & coord_mask[..., None, :]
    pair_mask = pair_mask & ~torch.eye(pred.shape[-2], device=pred.device, dtype=torch.bool)

    radius = torch.full_like(target_d, default_radius)
    if is_dna is not None or is_rna is not None:
        nucleotide = torch.zeros(coord_mask.shape, device=pred.device, dtype=torch.bool)
        for mask in [is_dna, is_rna]:
            if mask is None:
                continue
            mask = _ensure_batched_mask(mask.to(device=pred.device), batch_size)
            nucleotide = nucleotide | mask.to(dtype=torch.bool)
        nucleotide_pair = nucleotide[..., :, None] | nucleotide[..., None, :]
        radius = torch.where(nucleotide_pair, nucleotide_radius, radius)

    pair_mask = pair_mask & (target_d < radius)
    per_sample = (score * pair_mask).sum(dim=(-1, -2)) / pair_mask.sum(
        dim=(-1, -2)
    ).clamp_min(eps)
    return (1.0 - per_sample).mean()


def sequence_diffusion_loss(
    log_probs: torch.Tensor,
    true_seq: torch.Tensor,
    xt_seq: torch.Tensor,
    sequence_time: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Masked diffusion cross entropy over masked, reliable protein residues."""
    if log_probs.ndim == 2:
        log_probs = log_probs.unsqueeze(0)
    if true_seq.ndim == 1:
        true_seq = true_seq.unsqueeze(0)
    if xt_seq.ndim == 1:
        xt_seq = xt_seq.unsqueeze(0)
    if valid_mask.ndim == 1:
        valid_mask = valid_mask.unsqueeze(0)

    nll = F.nll_loss(
        log_probs.reshape(-1, log_probs.shape[-1]),
        true_seq.reshape(-1),
        reduction="none",
    ).reshape_as(true_seq)

    masked = xt_seq != true_seq
    loss_mask = masked & valid_mask.to(device=true_seq.device, dtype=torch.bool)
    per_sample = (nll * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)
    weighted = per_sample / sequence_time.clamp_min(eps)
    has_loss = loss_mask.any(dim=-1)
    if not has_loss.any():
        return log_probs.sum() * 0.0
    return weighted[has_loss].mean()


def _select_distogram_representatives(
    coords: torch.Tensor,
    coord_mask: torch.Tensor,
    rep_atom_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    coords = _ensure_batched_coords(coords)
    coord_mask = _ensure_batched_mask(coord_mask, coords.shape[0]).to(dtype=torch.bool)
    rep_atom_mask = rep_atom_mask.to(device=coords.device, dtype=torch.bool)

    if rep_atom_mask.ndim == 2:
        if not (rep_atom_mask == rep_atom_mask[0]).all():
            raise ValueError("Batched distogram representative masks must be identical for now.")
        rep_atom_mask = rep_atom_mask[0]

    return coords[:, rep_atom_mask], coord_mask[:, rep_atom_mask]


def distogram_training_loss(
    distogram_logits: torch.Tensor,
    target_coords: torch.Tensor,
    coord_mask: torch.Tensor,
    rep_atom_mask: torch.Tensor,
) -> torch.Tensor:
    """Wrapper around OpenFold distogram loss using DISCO representative atoms."""
    pseudo_beta, pseudo_beta_mask = _select_distogram_representatives(
        target_coords,
        coord_mask,
        rep_atom_mask,
    )
    return openfold_distogram_loss(
        logits=distogram_logits,
        pseudo_beta=pseudo_beta,
        pseudo_beta_mask=pseudo_beta_mask,
        no_bins=distogram_logits.shape[-1],
    )


def combine_losses(
    sequence_loss: torch.Tensor,
    mse_loss: torch.Tensor,
    smooth_lddt: torch.Tensor,
    distogram: torch.Tensor,
    weights: dict,
) -> LossBreakdown:
    total = (
        weights["sequence"] * sequence_loss
        + weights["mse"] * mse_loss
        + weights["smooth_lddt"] * smooth_lddt
        + weights["distogram"] * distogram
    )
    return LossBreakdown(
        total=total,
        sequence=sequence_loss,
        mse=mse_loss,
        smooth_lddt=smooth_lddt,
        distogram=distogram,
    )

