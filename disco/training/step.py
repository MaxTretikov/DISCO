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

import torch

from disco.training.losses import (
    atom_weights_from_features,
    combine_losses,
    distogram_training_loss,
    sequence_diffusion_loss,
    smooth_lddt_loss,
    weighted_aligned_mse_loss,
)
from disco.training.noising import make_noised_batch, remask_masked_reference_features


def _require(batch: dict, key: str):
    if key not in batch:
        raise KeyError(f"Training batch is missing required key '{key}'.")
    return batch[key]


def _get_true_seq(batch: dict, feature_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    if "true_prot_restype" in batch:
        return batch["true_prot_restype"]
    if "labels" in batch and "true_prot_restype" in batch["labels"]:
        return batch["labels"]["true_prot_restype"]
    if "true_prot_restype" in feature_dict:
        return feature_dict["true_prot_restype"]
    raise KeyError(
        "Training batches must provide protein residue token labels as "
        "'true_prot_restype' in the batch, labels, or input_feature_dict."
    )


def _get_label(batch: dict, feature_dict: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    labels = batch.get("labels", {})
    if key in labels:
        return labels[key]
    if key in batch:
        return batch[key]
    if key in feature_dict:
        return feature_dict[key]
    raise KeyError(f"Training batch is missing label '{key}'.")


def _valid_sequence_mask(
    feature_dict: dict[str, torch.Tensor],
    true_seq: torch.Tensor,
) -> torch.Tensor:
    valid = torch.ones_like(true_seq, dtype=torch.bool)
    true_seq_mask = feature_dict.get("true_prot_restype_mask")
    if true_seq_mask is not None:
        valid = valid & true_seq_mask.to(device=true_seq.device, dtype=torch.bool)

    token_valid = None
    for key, invert in [
        ("res_occ_cutoff_mask", False),
        ("res_is_resolved_mask", False),
        ("histag_mask", True),
    ]:
        if key not in feature_dict:
            continue
        mask = feature_dict[key].to(device=true_seq.device, dtype=torch.bool)
        mask = ~mask if invert else mask
        token_valid = mask if token_valid is None else token_valid & mask

    if token_valid is None:
        return valid

    prot_residue_mask = feature_dict.get("prot_residue_mask")
    if prot_residue_mask is None:
        if token_valid.ndim == 1 and valid.ndim == 2:
            token_valid = token_valid.unsqueeze(0).expand_as(valid)
        return valid & token_valid

    if token_valid.ndim == 1:
        token_valid = token_valid.unsqueeze(0)
    if prot_residue_mask.ndim == 1:
        prot_residue_mask = prot_residue_mask.unsqueeze(0)

    protein_valid = torch.zeros_like(valid)
    for batch_idx in range(valid.shape[0]):
        if token_valid.shape[-1] == prot_residue_mask.shape[-1]:
            sample_valid = token_valid[batch_idx][
                prot_residue_mask[batch_idx].to(device=true_seq.device, dtype=torch.bool)
            ]
        else:
            sample_valid = token_valid[batch_idx]
        n = min(sample_valid.shape[0], protein_valid.shape[-1])
        protein_valid[batch_idx, :n] = sample_valid[:n]
    return valid & protein_valid
    return valid


def compute_training_loss(
    model,
    batch: dict,
    loss_weights: dict,
    sequence_conditioning_mode: str = "scheduler",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Runs one DISCO training forward pass and returns total loss plus logs."""
    feature_dict = _require(batch, "input_feature_dict")
    if "true_prot_restype_mask" in batch:
        feature_dict = dict(feature_dict)
        feature_dict["true_prot_restype_mask"] = batch["true_prot_restype_mask"]
    labels = batch.get("labels", {})
    x0_struct = labels.get("coordinate", batch.get("coordinate"))
    coord_mask = labels.get("coordinate_mask", batch.get("coordinate_mask"))
    if x0_struct is None or coord_mask is None:
        raise KeyError("Training batch must include coordinate and coordinate_mask labels.")

    true_seq = _get_true_seq(batch, feature_dict).to(device=x0_struct.device).long()
    valid_seq_mask = _valid_sequence_mask(feature_dict, true_seq)

    noised = make_noised_batch(
        x0_struct=x0_struct,
        true_seq=true_seq,
        model=model,
        coord_mask=coord_mask,
        valid_seq_mask=valid_seq_mask,
        sequence_conditioning_mode=sequence_conditioning_mode,
    )

    feature_dict = dict(feature_dict)
    feature_dict["masked_prot_restype"] = noised.xt_seq
    feature_dict = remask_masked_reference_features(
        feature_dict,
        sequence_mask=noised.sequence_mask,
    )

    s_inputs, s, z, s_skip, z_skip, _, encoding_dict = model.get_pairformer_output(
        feature_dict,
        N_cycle=model.N_cycle,
        task_info={},
        sigma_seq=noised.sequence_conditioning,
        xt_noised_struct=noised.xt_struct,
        sigma=noised.structure_noise,
        inplace_safe=False,
        chunk_size=None,
    )

    pred_struct, seq_logits_pre = model.diffusion_module(
        x_noisy=noised.xt_struct,
        t_hat_noise_level_struct=noised.structure_noise,
        t_hat_noise_level_seq=noised.sequence_conditioning,
        input_feature_dict=feature_dict,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        s_skip=s_skip,
        z_skip=z_skip,
        encoding_dict=encoding_dict,
    )

    seq_log_probs = model.apply_subs_parameterization(
        seq_logits_pre,
        noised.xt_seq,
        exclude_unk=False,
        enforce_unmask_stay=False,
    )

    atom_weights = atom_weights_from_features(feature_dict, coord_mask)
    mse = weighted_aligned_mse_loss(
        pred=pred_struct,
        target=x0_struct,
        coord_mask=coord_mask,
        atom_weights=atom_weights,
    )
    slddt = smooth_lddt_loss(
        pred=pred_struct,
        target=x0_struct,
        coord_mask=coord_mask,
        is_dna=feature_dict.get("is_dna"),
        is_rna=feature_dict.get("is_rna"),
    )
    seq = sequence_diffusion_loss(
        log_probs=seq_log_probs,
        true_seq=true_seq,
        xt_seq=noised.xt_seq,
        sequence_time=noised.sequence_time,
        valid_mask=valid_seq_mask,
    )
    dist_logits = model.distogram_head(z)
    dist = distogram_training_loss(
        distogram_logits=dist_logits,
        target_coords=x0_struct,
        coord_mask=coord_mask,
        rep_atom_mask=_get_label(batch, feature_dict, "distogram_rep_atom_mask"),
        atom_to_token_idx=feature_dict.get("atom_to_token_idx"),
    )

    losses = combine_losses(
        sequence_loss=seq,
        mse_loss=mse,
        smooth_lddt=slddt,
        distogram=dist,
        weights=loss_weights,
    )
    log_dict = losses.as_log_dict()
    log_dict.update(
        {
            "noise/structure": noised.structure_noise.mean().detach(),
            "noise/sequence_time": noised.sequence_time.mean().detach(),
            "noise/sequence_conditioning": noised.sequence_conditioning.mean().detach(),
            "mask/sequence_fraction": noised.sequence_mask.float().mean().detach(),
        }
    )
    return losses.total, log_dict
