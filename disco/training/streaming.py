# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from disco.training.data import _validate_example, training_collate_fn
from disco.training.pdb_dataset import (
    _BETA,
    _choose_crop_mode,
    _read_protenix_index,
    _structure_path_for_entry,
    build_sample_records_from_protenix_index,
    crop_atom_array_for_sample,
)
from disco.training.preprocess import (
    atom_array_to_training_example,
    prepare_training_atom_array,
)


@dataclass(frozen=True)
class StreamingProtenixRow:
    row_index: int
    entry_id: str
    index_row: Any
    sampling_weight: float


def _cache_limit_bytes(max_cache_gb: float | None) -> int | None:
    if max_cache_gb is None or max_cache_gb <= 0:
        return None
    return int(max_cache_gb * 1024**3)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


class StreamingProtenixDataset(Dataset):
    """On-the-fly Protenix training dataset with an LRU disk cache.

    The dataset stores only generated crops in ``cache_dir``. Cache entries are
    keyed by Protenix index row plus crop settings, and old entries are removed
    when the configured byte budget is exceeded.
    """

    def __init__(
        self,
        archive_root: str | Path = "/mnt/archive/datasets/protenix",
        protenix_index: str | Path = (
            "/mnt/archive/datasets/protenix/indices/"
            "indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
        ),
        cache_dir: str | Path = "/mnt/archive/datasets/disco_training/cache/protenix",
        *,
        cutoff_date: str | None = "2021-09-30",
        crop_size: int = 384,
        crop_mode: str = "weighted",
        max_entries: int | None = None,
        max_rows: int | None = None,
        seed: int = 0,
        bb_only: bool = True,
        validate: bool = True,
        max_cache_gb: float | None = 256.0,
        cache_cleanup_interval: int = 128,
        drop_bond_mask: bool = True,
        suppress_parser_warnings: bool = True,
    ) -> None:
        self.archive_root = Path(archive_root)
        self.protenix_index = Path(protenix_index)
        self.cache_dir = Path(cache_dir)
        self.crop_size = crop_size
        self.crop_mode = crop_mode
        self.seed = seed
        self.bb_only = bb_only
        self.validate = validate
        self.max_cache_bytes = _cache_limit_bytes(max_cache_gb)
        self.cache_cleanup_interval = max(cache_cleanup_interval, 1)
        self.drop_bond_mask = drop_bond_mask
        self.suppress_parser_warnings = suppress_parser_warnings
        self._getitem_count = 0

        cutoff = date.fromisoformat(cutoff_date) if cutoff_date else None
        entry_rows, cluster_counts = _read_protenix_index(self.protenix_index, cutoff)
        entry_ids = sorted(entry_rows)
        if max_entries is not None:
            entry_ids = entry_ids[:max_entries]

        rows = []
        for entry_id in entry_ids:
            for index_row in entry_rows[entry_id]:
                cluster_size = cluster_counts.get(index_row.cluster_id, 1)
                weight = _BETA[index_row.sample_kind] * max(index_row.approx_tokens, 1)
                weight = weight / max(cluster_size, 1)
                rows.append(
                    StreamingProtenixRow(
                        row_index=len(rows),
                        entry_id=entry_id,
                        index_row=index_row,
                        sampling_weight=max(float(weight), 0.0),
                    )
                )
                if max_rows is not None and len(rows) >= max_rows:
                    break
            if max_rows is not None and len(rows) >= max_rows:
                break

        if len(rows) == 0:
            raise ValueError(f"No Protenix rows found in index: {self.protenix_index}")

        self.rows = rows
        self.sampling_weights = torch.tensor(
            [row.sampling_weight for row in rows],
            dtype=torch.double,
        ).clamp_min(0.0)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_exc: Exception | None = None
        for offset in range(min(8, len(self.rows))):
            row_idx = (idx + offset) % len(self.rows)
            try:
                example = self._get_or_build_example(row_idx)
                if self.validate:
                    _validate_example(example, self._cache_path(row_idx))
                return example
            except Exception as exc:
                last_exc = exc
                logging.getLogger(__name__).warning(
                    "Skipping Protenix row %d after error: %r",
                    row_idx,
                    exc,
                )
        raise RuntimeError(
            f"Could not build a valid Protenix example near row {idx}"
        ) from last_exc

    def _cache_key(self, idx: int) -> str:
        row = self.rows[idx]
        payload = {
            "row_index": row.row_index,
            "entry_id": row.entry_id,
            "kind": row.index_row.sample_kind,
            "chains": row.index_row.chain_ids,
            "cluster": row.index_row.cluster_id,
            "crop_size": self.crop_size,
            "crop_mode": self.crop_mode,
            "seed": self.seed,
            "bb_only": self.bb_only,
            "drop_bond_mask": self.drop_bond_mask,
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha1(encoded).hexdigest()

    def _cache_path(self, idx: int) -> Path:
        key = self._cache_key(idx)
        return self.cache_dir / key[:2] / f"{key}.pt"

    def _get_or_build_example(self, idx: int) -> dict[str, Any]:
        cache_path = self._cache_path(idx)
        if cache_path.exists():
            os.utime(cache_path, None)
            return torch.load(cache_path, map_location="cpu", weights_only=False)

        example = self._build_example(idx)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(f".tmp.{os.getpid()}")
        torch.save(example, tmp_path)
        tmp_path.replace(cache_path)
        self._getitem_count += 1
        if self._getitem_count % self.cache_cleanup_interval == 0:
            self._prune_cache()
        return example

    def _build_example(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        structure_path = _structure_path_for_entry(
            self.archive_root,
            "protenix",
            row.entry_id,
        )
        parser_logger = logging.getLogger("disco.data.parser")
        old_parser_level = parser_logger.level
        if self.suppress_parser_warnings:
            parser_logger.setLevel(logging.ERROR)
        try:
            atom_array, _ = prepare_training_atom_array(structure_path, bb_only=self.bb_only)
        finally:
            parser_logger.setLevel(old_parser_level)

        records = build_sample_records_from_protenix_index(
            atom_array,
            structure_path,
            row.entry_id,
            [row.index_row],
            {row.index_row.cluster_id: 1},
        )
        records = [record for record in records if record.weight > 0 and record.n_prot > 0]
        if not records:
            raise ValueError(f"Protenix row produced no protein training record: {row}")

        record = records[0]
        rng = np.random.default_rng(self.seed + row.row_index)
        crop_mode = _choose_crop_mode(record.sample_kind, self.crop_mode, rng)
        cropped = crop_atom_array_for_sample(
            atom_array,
            record,
            crop_size=self.crop_size,
            crop_mode=crop_mode,
            rng=rng,
        )
        example = atom_array_to_training_example(cropped)
        if self.drop_bond_mask:
            example["input_feature_dict"].pop("bond_mask", None)
        example["sample_weight"] = torch.tensor(row.sampling_weight, dtype=torch.float32)
        return example

    def _prune_cache(self) -> None:
        if self.max_cache_bytes is None:
            return

        files = [path for path in self.cache_dir.glob("*/*.pt") if path.is_file()]
        total = sum(_file_size(path) for path in files)
        if total <= self.max_cache_bytes:
            return

        files.sort(key=lambda path: path.stat().st_mtime)
        for path in files:
            total -= _file_size(path)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            if total <= self.max_cache_bytes:
                break


def create_streaming_protenix_dataloader(
    manifest_path: str | Path | None = None,
    archive_root: str | Path = "/mnt/archive/datasets/protenix",
    protenix_index: str | Path = (
        "/mnt/archive/datasets/protenix/indices/"
        "indices_20260107-20chains_before_2021-09-30_res4.5.csv.gz"
    ),
    cache_dir: str | Path = "/mnt/archive/datasets/disco_training/cache/protenix",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
    weighted_sampling: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    del manifest_path
    dataset = StreamingProtenixDataset(
        archive_root=archive_root,
        protenix_index=protenix_index,
        cache_dir=cache_dir,
        **dataset_kwargs,
    )
    sampler = None
    if weighted_sampling:
        if dataset.sampling_weights.sum() <= 0:
            raise ValueError("Cannot use weighted sampling with all-zero Protenix weights.")
        sampler = WeightedRandomSampler(
            weights=dataset.sampling_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=training_collate_fn,
    )
