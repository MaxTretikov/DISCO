# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


_REQUIRED_TOP_LEVEL = (
    "input_feature_dict",
    "coordinate",
    "coordinate_mask",
    "true_prot_restype",
)

_REQUIRED_FEATURES = (
    "ref_pos",
    "ref_charge",
    "ref_mask",
    "ref_element",
    "ref_atom_name_chars",
    "asym_id",
    "residue_index",
    "entity_id",
    "sym_id",
    "token_index",
    "token_bonds",
    "atom_to_token_idx",
    "prot_residue_mask",
    "distogram_rep_atom_mask",
)


def _read_manifest(manifest_path: Path) -> list[Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Training manifest does not exist: {manifest_path}")

    if manifest_path.suffix == ".json":
        raw = json.loads(manifest_path.read_text())
        if not isinstance(raw, list):
            raise ValueError("JSON training manifest must be a list of paths or objects.")
        entries = [item["path"] if isinstance(item, dict) else item for item in raw]
    elif manifest_path.suffix == ".jsonl":
        entries = []
        for line in manifest_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            entries.append(item["path"] if isinstance(item, dict) else item)
    else:
        entries = [
            line.strip()
            for line in manifest_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    base_dir = manifest_path.parent
    paths = [Path(entry) for entry in entries]
    return [path if path.is_absolute() else base_dir / path for path in paths]


def _get_label(example: Mapping[str, Any], key: str):
    labels = example.get("labels", {})
    if isinstance(labels, Mapping) and key in labels:
        return labels[key]
    return example.get(key)


def _validate_example(example: Mapping[str, Any], path: Path) -> None:
    for key in _REQUIRED_TOP_LEVEL:
        if key == "coordinate" and _get_label(example, key) is not None:
            continue
        if key == "coordinate_mask" and _get_label(example, key) is not None:
            continue
        if key == "true_prot_restype" and _get_label(example, key) is not None:
            continue
        if key not in example:
            raise KeyError(f"{path} is missing required key '{key}'.")

    feature_dict = example["input_feature_dict"]
    if not isinstance(feature_dict, Mapping):
        raise TypeError(f"{path}['input_feature_dict'] must be a mapping.")

    for key in _REQUIRED_FEATURES:
        if key not in feature_dict and key not in example:
            raise KeyError(f"{path} is missing required feature '{key}'.")

    coordinate = _get_label(example, "coordinate")
    coordinate_mask = _get_label(example, "coordinate_mask")
    true_seq = _get_label(example, "true_prot_restype")
    if not isinstance(coordinate, torch.Tensor) or coordinate.ndim != 2:
        raise ValueError(f"{path} coordinate must be a tensor with shape [N_atom, 3].")
    if coordinate.shape[-1] != 3:
        raise ValueError(f"{path} coordinate must have final dimension 3.")
    if not isinstance(coordinate_mask, torch.Tensor) or coordinate_mask.ndim != 1:
        raise ValueError(f"{path} coordinate_mask must be a tensor with shape [N_atom].")
    if coordinate.shape[0] != coordinate_mask.shape[0]:
        raise ValueError(f"{path} coordinate and coordinate_mask lengths differ.")
    if not isinstance(true_seq, torch.Tensor) or true_seq.ndim != 1:
        raise ValueError(f"{path} true_prot_restype must be a tensor with shape [N_prot].")


def _collate_values(values: Sequence[Any], key_path: str) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        shapes = [tuple(value.shape) for value in values]
        if len(set(shapes)) != 1:
            raise ValueError(
                f"Cannot batch variable-shape tensors at {key_path}: {shapes}. "
                "Use batch_size=1 until crop shapes are fixed."
            )
        return torch.stack(list(values), dim=0)

    if isinstance(first, Mapping):
        keys = first.keys()
        return {
            key: _collate_values([value[key] for value in values], f"{key_path}.{key}")
            for key in keys
            if all(key in value for value in values)
        }

    if first is None:
        return None

    return list(values)


def training_collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates preprocessed training examples.

    Tensors are stacked only when shapes match. Non-tensor metadata is kept as a
    list. This makes batch size 1 robust for early dataset work while still
    allowing fixed-shape crops to batch later.
    """
    if len(examples) == 0:
        raise ValueError("Cannot collate an empty batch.")
    return _collate_values(examples, "batch")


class PreprocessedTrainingDataset(Dataset):
    """Dataset for already-featurized DISCO training examples.

    Each manifest entry should point to a ``torch.save``-serialized dictionary
    with the current training batch contract:

    - ``input_feature_dict``
    - ``coordinate`` or ``labels.coordinate``
    - ``coordinate_mask`` or ``labels.coordinate_mask``
    - ``true_prot_restype`` or ``labels.true_prot_restype``
    - ``distogram_rep_atom_mask`` in either the example or feature dict
    """

    def __init__(self, manifest_path: str | Path, validate: bool = True) -> None:
        self.manifest_path = Path(manifest_path)
        self.paths = _read_manifest(self.manifest_path)
        if len(self.paths) == 0:
            raise ValueError(f"Training manifest is empty: {self.manifest_path}")
        self.validate = validate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        example = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(example, Mapping):
            raise TypeError(f"Training example must be a mapping: {path}")
        example = dict(example)
        if self.validate:
            _validate_example(example, path)
        return example


def create_preprocessed_dataloader(
    manifest_path: str | Path | None,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
    validate: bool = True,
) -> DataLoader:
    """Creates a dataloader for manifest-listed preprocessed examples."""
    if manifest_path is None:
        raise ValueError(
            "`training.dataloader.manifest_path` must point to a manifest of "
            "preprocessed `.pt` training examples."
        )

    dataset = PreprocessedTrainingDataset(manifest_path=manifest_path, validate=validate)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=training_collate_fn,
    )
