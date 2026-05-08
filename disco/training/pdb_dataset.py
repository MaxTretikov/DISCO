# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from disco.data.tokenizer import AtomArrayTokenizer
from disco.training.preprocess import (
    atom_array_to_training_example,
    prepare_training_atom_array,
)


_CROP_MODE_WEIGHTS = {
    "contiguous": 0.2,
    "spatial": 0.4,
    "interface": 0.4,
}
_BETA = {
    "chain": 0.5,
    "interface": 1.0,
}
_ALPHA_PROT = 3.0
_ALPHA_NUC = 3.0
_ALPHA_LIGAND = 1.0
_INTERFACE_DISTANCE_A = 5.0
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class SampleRecord:
    entry_id: str
    structure_path: str
    sample_kind: str
    chain_ids: tuple[str, ...]
    selected_token_indices: tuple[int, ...]
    n_prot: int
    n_nuc: int
    n_ligand: int
    cluster_id: str
    cluster_size: int
    weight: float
    deposition_date: str | None = None
    crop_mode: str | None = None
    output_path: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ProtenixIndexRow:
    sample_kind: str
    chain_ids: tuple[str, ...]
    cluster_id: str
    deposition_date: str | None


def _entry_id_from_path(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".cif"):
        name = name[:-4]
    if ".pdb" in name:
        name = name.split(".pdb", 1)[0]
    return name.lower()


def _structure_paths(archive_root: Path, source: str) -> list[Path]:
    if source == "biounit":
        root = archive_root / "pdb" / "biounit"
        pattern = "*.pdb*.gz"
    elif source == "mmcif":
        root = archive_root / "pdb" / "mmcif"
        pattern = "*.cif.gz"
    elif source == "protenix":
        root = archive_root / "mmcif"
        pattern = "*.cif"
    else:
        raise ValueError(f"Unknown structure source: {source}")
    if source == "protenix":
        return sorted(root.glob(pattern))
    return sorted(root.glob(f"*/{pattern}"))


def _structure_path_for_entry(archive_root: Path, source: str, entry_id: str) -> Path:
    if source == "biounit":
        candidates = sorted(
            (archive_root / "pdb" / "biounit" / entry_id[1:3]).glob(
                f"{entry_id}.pdb*.gz"
            )
        )
        if not candidates:
            candidates = sorted(
                (archive_root / "pdb" / "biounit" / entry_id[1:3]).glob(
                    f"{entry_id.lower()}.pdb*.gz"
                )
            )
        if not candidates:
            return archive_root / "pdb" / "biounit" / entry_id[1:3] / f"{entry_id}.pdb1.gz"
        return candidates[0]
    if source == "mmcif":
        return archive_root / "pdb" / "mmcif" / entry_id[1:3] / f"{entry_id}.cif.gz"
    if source == "protenix":
        return archive_root / "mmcif" / f"{entry_id}.cif"
    raise ValueError(f"Unknown structure source: {source}")


def _mmcif_path_for_entry(archive_root: Path, entry_id: str) -> Path:
    return archive_root / "pdb" / "mmcif" / entry_id[1:3] / f"{entry_id}.cif.gz"


def _read_mmcif_deposition_date(path: Path) -> str | None:
    if not path.exists():
        return None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("_pdbx_database_status.recvd_initial_deposition_date"):
                fields = line.split()
                if len(fields) >= 2 and _DATE_RE.match(fields[-1]):
                    return fields[-1]
    return None


def _passes_cutoff(deposition_date: str | None, cutoff: date | None) -> bool:
    if cutoff is None or deposition_date is None:
        return True
    return date.fromisoformat(deposition_date) <= cutoff


def _read_seqres_clusters(seqres_path: Path | None) -> dict[str, tuple[str, int]]:
    if seqres_path is None or not seqres_path.exists():
        return {}

    opener = gzip.open if seqres_path.suffix == ".gz" else open
    chain_to_sequence = {}
    current_key = None
    with opener(seqres_path, "rt", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                current_key = line[1:].split()[0].lower()
                continue
            if current_key is not None:
                chain_to_sequence[current_key] = line
                current_key = None

    sequence_counts: dict[str, int] = {}
    for sequence in chain_to_sequence.values():
        sequence_counts[sequence] = sequence_counts.get(sequence, 0) + 1

    return {
        key: (sequence, sequence_counts[sequence])
        for key, sequence in chain_to_sequence.items()
    }


def _read_cluster_file(cluster_file: Path | None) -> dict[str, tuple[str, int]]:
    if cluster_file is None:
        return {}

    member_to_cluster: dict[str, str] = {}
    with cluster_file.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\t, ]+", line)
            if len(parts) < 2:
                continue
            member, cluster = parts[0].lower(), parts[1].lower()
            member_to_cluster[member] = cluster

    cluster_counts: dict[str, int] = {}
    for cluster in member_to_cluster.values():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    return {
        member: (cluster, cluster_counts[cluster])
        for member, cluster in member_to_cluster.items()
    }


def _open_text(path: Path):
    return (
        gzip.open(path, "rt", errors="replace")
        if path.suffix == ".gz"
        else path.open("rt", errors="replace")
    )


def _read_protenix_index(
    index_path: Path | None,
    cutoff: date | None,
) -> tuple[dict[str, list[ProtenixIndexRow]], dict[str, int]]:
    if index_path is None:
        return {}, {}

    entry_rows: dict[str, list[ProtenixIndexRow]] = {}
    cluster_counts: dict[str, int] = {}
    with _open_text(index_path) as handle:
        reader = csv.DictReader(handle)
        required = {"pdb_id", "type", "chain_1_id", "cluster_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Protenix index {index_path} is missing columns: {sorted(missing)}"
            )

        for row in reader:
            sample_kind = (row.get("type") or "").strip()
            if sample_kind not in _BETA:
                continue

            mol_types = {
                (row.get("mol_1_type") or "").strip(),
                (row.get("mol_2_type") or "").strip(),
            }
            if "prot" not in mol_types:
                continue

            deposition_date = (row.get("release_date") or "").strip() or None
            if not _passes_cutoff(deposition_date, cutoff):
                continue

            entry_id = (row.get("pdb_id") or "").strip().lower()
            if not entry_id:
                continue

            chain_ids = tuple(
                chain_id
                for chain_id in (
                    (row.get("chain_1_id") or "").strip(),
                    (row.get("chain_2_id") or "").strip(),
                )
                if chain_id
            )
            if len(chain_ids) == 0:
                continue

            cluster_id = (row.get("cluster_id") or "").strip().lower()
            if not cluster_id:
                cluster_parts = [
                    (row.get("cluster_1_id") or "").strip().lower(),
                    (row.get("cluster_2_id") or "").strip().lower(),
                ]
                cluster_id = "+".join(sorted(part for part in cluster_parts if part))
            if not cluster_id:
                cluster_id = "+".join(f"{entry_id}_{chain_id}".lower() for chain_id in chain_ids)

            index_row = ProtenixIndexRow(
                sample_kind=sample_kind,
                chain_ids=chain_ids,
                cluster_id=cluster_id,
                deposition_date=deposition_date,
            )
            entry_rows.setdefault(entry_id, []).append(index_row)
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    return entry_rows, cluster_counts


def _token_metadata(atom_array):
    token_array = AtomArrayTokenizer(atom_array).get_token_array()
    centre_indices = np.asarray(token_array.get_annotation("centre_atom_index"))
    centre_atoms = atom_array[centre_indices]
    token_chain_ids = centre_atoms.chain_id.astype(str)
    token_mol_types = centre_atoms.mol_type.astype(str)
    token_centres = np.asarray(centre_atoms.coord, dtype=np.float32)
    return token_array, token_chain_ids, token_mol_types, token_centres


def _count_token_types(
    token_mol_types: np.ndarray,
    token_indices: np.ndarray,
) -> tuple[int, int, int]:
    mol_types = token_mol_types[token_indices]
    n_prot = int(np.count_nonzero(mol_types == "protein"))
    n_nuc = int(np.count_nonzero((mol_types == "dna") | (mol_types == "rna")))
    n_ligand = int(np.count_nonzero(mol_types == "ligand"))
    return n_prot, n_nuc, n_ligand


def _cluster_for_chains(
    entry_id: str,
    chain_ids: Iterable[str],
    cluster_map: dict[str, tuple[str, int]],
) -> tuple[str, int]:
    clusters = []
    sizes = []
    for chain_id in chain_ids:
        key = f"{entry_id}_{chain_id.removesuffix('0')}".lower()
        cluster_id, cluster_size = cluster_map.get(key, (key, 1))
        clusters.append(cluster_id)
        sizes.append(cluster_size)
    return "+".join(sorted(clusters)), max(sizes, default=1)


def _token_mask_for_index_chains(
    token_chain_ids: np.ndarray,
    index_chain_ids: tuple[str, ...],
) -> np.ndarray:
    token_mask = np.zeros(token_chain_ids.shape, dtype=bool)
    for index_chain_id in index_chain_ids:
        token_mask |= token_chain_ids == index_chain_id
        suffix_mask = np.fromiter(
            (
                token_chain_id.startswith(index_chain_id)
                and token_chain_id[len(index_chain_id) :].isdigit()
                for token_chain_id in token_chain_ids
            ),
            dtype=bool,
            count=len(token_chain_ids),
        )
        token_mask |= suffix_mask
    return token_mask


def _sample_weight(
    sample_kind: str,
    n_prot: int,
    n_nuc: int,
    n_ligand: int,
    cluster_size: int,
) -> float:
    score = _ALPHA_PROT * n_prot + _ALPHA_NUC * n_nuc + _ALPHA_LIGAND * n_ligand
    if score <= 0:
        return 0.0
    return _BETA[sample_kind] * score / max(cluster_size, 1)


def _min_inter_chain_distance(atom_array, chain_a: str, chain_b: str) -> float:
    coords_a = np.asarray(atom_array.coord[atom_array.chain_id.astype(str) == chain_a])
    coords_b = np.asarray(atom_array.coord[atom_array.chain_id.astype(str) == chain_b])
    if len(coords_a) == 0 or len(coords_b) == 0:
        return math.inf
    if np.any(coords_a.min(axis=0) - coords_b.max(axis=0) > _INTERFACE_DISTANCE_A):
        return math.inf
    if np.any(coords_b.min(axis=0) - coords_a.max(axis=0) > _INTERFACE_DISTANCE_A):
        return math.inf
    distances = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=-1)
    return float(distances.min())


def build_sample_records(
    atom_array,
    structure_path: Path,
    entry_id: str,
    cluster_map: dict[str, tuple[str, int]],
    deposition_date: str | None,
) -> list[SampleRecord]:
    _, token_chain_ids, token_mol_types, _ = _token_metadata(atom_array)
    polymer_token_mask = np.isin(token_mol_types, ["protein", "dna", "rna"])
    polymer_chain_ids = sorted(set(token_chain_ids[polymer_token_mask]))
    records = []

    for chain_id in polymer_chain_ids:
        token_indices = np.nonzero(token_chain_ids == chain_id)[0]
        n_prot, n_nuc, n_ligand = _count_token_types(token_mol_types, token_indices)
        if n_prot + n_nuc == 0:
            continue
        cluster_id, cluster_size = _cluster_for_chains(entry_id, [chain_id], cluster_map)
        weight = _sample_weight("chain", n_prot, n_nuc, n_ligand, cluster_size)
        records.append(
            SampleRecord(
                entry_id=entry_id,
                structure_path=str(structure_path),
                sample_kind="chain",
                chain_ids=(chain_id,),
                selected_token_indices=tuple(int(i) for i in token_indices),
                n_prot=n_prot,
                n_nuc=n_nuc,
                n_ligand=n_ligand,
                cluster_id=cluster_id,
                cluster_size=cluster_size,
                weight=weight,
                deposition_date=deposition_date,
            )
        )

    for i, chain_a in enumerate(polymer_chain_ids):
        for chain_b in polymer_chain_ids[i + 1 :]:
            if _min_inter_chain_distance(atom_array, chain_a, chain_b) >= _INTERFACE_DISTANCE_A:
                continue
            token_indices = np.nonzero(
                (token_chain_ids == chain_a) | (token_chain_ids == chain_b)
            )[0]
            n_prot, n_nuc, n_ligand = _count_token_types(token_mol_types, token_indices)
            cluster_id, cluster_size = _cluster_for_chains(
                entry_id,
                [chain_a, chain_b],
                cluster_map,
            )
            weight = _sample_weight("interface", n_prot, n_nuc, n_ligand, cluster_size)
            records.append(
                SampleRecord(
                    entry_id=entry_id,
                    structure_path=str(structure_path),
                    sample_kind="interface",
                    chain_ids=(chain_a, chain_b),
                    selected_token_indices=tuple(int(i) for i in token_indices),
                    n_prot=n_prot,
                    n_nuc=n_nuc,
                    n_ligand=n_ligand,
                    cluster_id=cluster_id,
                    cluster_size=cluster_size,
                    weight=weight,
                    deposition_date=deposition_date,
                )
            )

    return records


def build_sample_records_from_protenix_index(
    atom_array,
    structure_path: Path,
    entry_id: str,
    index_rows: list[ProtenixIndexRow],
    cluster_counts: dict[str, int],
) -> list[SampleRecord]:
    _, token_chain_ids, token_mol_types, _ = _token_metadata(atom_array)
    records = []

    for index_row in index_rows:
        token_mask = _token_mask_for_index_chains(token_chain_ids, index_row.chain_ids)
        token_indices = np.nonzero(token_mask)[0]
        if len(token_indices) == 0:
            continue

        n_prot, n_nuc, n_ligand = _count_token_types(token_mol_types, token_indices)
        cluster_size = cluster_counts.get(index_row.cluster_id, 1)
        weight = _sample_weight(
            index_row.sample_kind,
            n_prot,
            n_nuc,
            n_ligand,
            cluster_size,
        )
        records.append(
            SampleRecord(
                entry_id=entry_id,
                structure_path=str(structure_path),
                sample_kind=index_row.sample_kind,
                chain_ids=index_row.chain_ids,
                selected_token_indices=tuple(int(i) for i in token_indices),
                n_prot=n_prot,
                n_nuc=n_nuc,
                n_ligand=n_ligand,
                cluster_id=index_row.cluster_id,
                cluster_size=cluster_size,
                weight=weight,
                deposition_date=index_row.deposition_date,
            )
        )

    return records


def _choose_crop_mode(sample_kind: str, requested_mode: str, rng: np.random.Generator) -> str:
    if requested_mode == "interface" and sample_kind != "interface":
        return "spatial"
    if requested_mode != "weighted":
        return requested_mode
    modes = list(_CROP_MODE_WEIGHTS)
    probs = np.asarray([_CROP_MODE_WEIGHTS[mode] for mode in modes], dtype=np.float64)
    if sample_kind != "interface":
        probs[modes.index("interface")] = 0.0
    probs = probs / probs.sum()
    return str(rng.choice(modes, p=probs))


def _token_distances(token_centres: np.ndarray, seed_indices: np.ndarray) -> np.ndarray:
    seed_centres = token_centres[seed_indices]
    distances = np.linalg.norm(
        token_centres[:, None, :] - seed_centres[None, :, :],
        axis=-1,
    )
    return distances.min(axis=1)


def crop_atom_array_for_sample(
    atom_array,
    record: SampleRecord,
    crop_size: int,
    crop_mode: str,
    rng: np.random.Generator,
):
    token_array, token_chain_ids, _, token_centres = _token_metadata(atom_array)
    selected = np.asarray(record.selected_token_indices, dtype=np.int64)
    if len(token_array) <= crop_size:
        return atom_array

    if crop_mode == "contiguous":
        if len(selected) <= crop_size:
            selected_crop = selected
        else:
            start = int(rng.integers(0, len(selected) - crop_size + 1))
            selected_crop = selected[start : start + crop_size]
        distances = _token_distances(token_centres, selected_crop)
        crop_tokens = np.argsort(distances, kind="stable")[:crop_size]
    elif crop_mode == "spatial":
        seed = np.asarray([int(rng.choice(selected))], dtype=np.int64)
        distances = _token_distances(token_centres, seed)
        crop_tokens = np.argsort(distances, kind="stable")[:crop_size]
    elif crop_mode == "interface":
        if len(record.chain_ids) < 2:
            return crop_atom_array_for_sample(atom_array, record, crop_size, "spatial", rng)
        chain_a, chain_b = record.chain_ids[:2]
        tokens_a = np.nonzero(token_chain_ids == chain_a)[0]
        tokens_b = np.nonzero(token_chain_ids == chain_b)[0]
        if len(tokens_a) == 0 or len(tokens_b) == 0:
            return crop_atom_array_for_sample(atom_array, record, crop_size, "spatial", rng)
        distances = np.linalg.norm(
            token_centres[tokens_a][:, None, :] - token_centres[tokens_b][None, :, :],
            axis=-1,
        )
        pair_idx = np.unravel_index(int(distances.argmin()), distances.shape)
        seeds = np.asarray([tokens_a[pair_idx[0]], tokens_b[pair_idx[1]]], dtype=np.int64)
        crop_tokens = np.argsort(_token_distances(token_centres, seeds), kind="stable")[:crop_size]
    else:
        raise ValueError(f"Unknown crop mode: {crop_mode}")

    crop_tokens = np.sort(crop_tokens)
    atom_indices = [
        atom_index
        for token_idx in crop_tokens
        for atom_index in token_array[int(token_idx)].atom_indices
    ]
    return atom_array[atom_indices]


def _relative_manifest_path(path: Path, manifest_path: Path) -> str:
    return str(
        path.relative_to(manifest_path.parent)
        if path.is_relative_to(manifest_path.parent)
        else path
    )


def _record_output_path(output_dir: Path, record: SampleRecord, sample_index: int) -> Path:
    assembly = Path(record.structure_path).name.replace(".gz", "").replace(".", "_")
    chain_tag = "_".join(record.chain_ids).replace("/", "-")
    filename = (
        f"{sample_index:08d}_{record.entry_id}_{assembly}_"
        f"{record.sample_kind}_{chain_tag}.pt"
    )
    return output_dir / record.entry_id[1:3] / filename


def process_pdb_archive(
    archive_root: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    source: str = "biounit",
    cutoff_date: str | None = "2021-09-30",
    crop_size: int = 384,
    crop_mode: str = "weighted",
    max_entries: int | None = None,
    samples_per_entry: int = 1,
    seed: int = 0,
    bb_only: bool = True,
    cluster_file: str | Path | None = None,
    seqres_path: str | Path | None = None,
    protenix_index: str | Path | None = None,
    suppress_parser_warnings: bool = True,
) -> Path:
    archive_root = Path(archive_root)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    metadata_path = Path(metadata_path) if metadata_path is not None else None
    cutoff = date.fromisoformat(cutoff_date) if cutoff_date else None

    if seqres_path is None:
        seqres_path = archive_root / "derived" / "pdb_seqres.txt.gz"
    cluster_map = _read_seqres_clusters(Path(seqres_path))
    cluster_map.update(_read_cluster_file(Path(cluster_file) if cluster_file else None))
    protenix_rows, protenix_cluster_counts = _read_protenix_index(
        Path(protenix_index) if protenix_index else None,
        cutoff,
    )

    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

    parser_logger = logging.getLogger("disco.data.parser")
    old_parser_level = parser_logger.level
    if suppress_parser_warnings:
        parser_logger.setLevel(logging.ERROR)

    if protenix_rows:
        entry_ids = sorted(protenix_rows)
        if max_entries is not None:
            entry_ids = entry_ids[:max_entries]
        structure_items = [
            (
                _structure_path_for_entry(archive_root, source, entry_id),
                entry_id,
                protenix_rows[entry_id],
            )
            for entry_id in entry_ids
        ]
    else:
        structure_paths = _structure_paths(archive_root, source)
        if max_entries is not None:
            structure_paths = structure_paths[:max_entries]
        structure_items = [
            (structure_path, _entry_id_from_path(structure_path), None)
            for structure_path in structure_paths
        ]

    try:
        sample_index = 0
        with manifest_path.open("w") as manifest_handle:
            metadata_handle = metadata_path.open("w") if metadata_path is not None else None
            try:
                for structure_path, entry_id, index_rows in structure_items:
                    if not structure_path.exists():
                        if metadata_handle is not None:
                            metadata_handle.write(
                                json.dumps(
                                    {
                                        "entry_id": entry_id,
                                        "structure_path": str(structure_path),
                                        "error": "missing structure file",
                                    }
                                )
                                + "\n"
                            )
                        continue
                    deposition_path = (
                        structure_path
                        if source == "protenix"
                        else _mmcif_path_for_entry(archive_root, entry_id)
                    )
                    deposition_date = _read_mmcif_deposition_date(deposition_path)
                    if index_rows is None and not _passes_cutoff(deposition_date, cutoff):
                        continue

                    try:
                        atom_array, _ = prepare_training_atom_array(
                            structure_path,
                            bb_only=bb_only,
                        )
                        if index_rows is None:
                            records = build_sample_records(
                                atom_array,
                                structure_path,
                                entry_id,
                                cluster_map,
                                deposition_date,
                            )
                        else:
                            records = build_sample_records_from_protenix_index(
                                atom_array,
                                structure_path,
                                entry_id,
                                index_rows,
                                protenix_cluster_counts,
                            )
                        records = [
                            record
                            for record in records
                            if record.weight > 0 and record.n_prot > 0
                        ]
                        if len(records) == 0:
                            continue

                        weights = np.asarray(
                            [record.weight for record in records],
                            dtype=np.float64,
                        )
                        weights = weights / weights.sum()
                        replace = len(records) < samples_per_entry
                        chosen = rng.choice(
                            len(records),
                            size=samples_per_entry,
                            replace=replace,
                            p=weights,
                        )

                        for record_idx in np.atleast_1d(chosen):
                            record = records[int(record_idx)]
                            selected_crop_mode = _choose_crop_mode(
                                record.sample_kind,
                                crop_mode,
                                rng,
                            )
                            cropped = crop_atom_array_for_sample(
                                atom_array,
                                record,
                                crop_size=crop_size,
                                crop_mode=selected_crop_mode,
                                rng=rng,
                            )
                            example = atom_array_to_training_example(cropped)
                            output_path = _record_output_path(
                                output_dir,
                                record,
                                sample_index,
                            )
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            torch.save(example, output_path)

                            record_dict = asdict(record)
                            record_dict["crop_mode"] = selected_crop_mode
                            record_dict["output_path"] = str(output_path)
                            manifest_record = {
                                "path": _relative_manifest_path(
                                    output_path,
                                    manifest_path,
                                ),
                                "weight": record.weight,
                            }
                            if manifest_path.suffix == ".jsonl":
                                manifest_handle.write(json.dumps(manifest_record) + "\n")
                            else:
                                manifest_handle.write(manifest_record["path"] + "\n")
                            if metadata_handle is not None:
                                metadata_handle.write(json.dumps(record_dict) + "\n")
                            sample_index += 1
                    except Exception as exc:
                        if metadata_handle is not None:
                            metadata_handle.write(
                                json.dumps(
                                    {
                                        "entry_id": entry_id,
                                        "structure_path": str(structure_path),
                                        "error": repr(exc),
                                    }
                                )
                                + "\n"
                            )
            finally:
                if metadata_handle is not None:
                    metadata_handle.close()

        if sample_index == 0:
            raise ValueError(
                "No training examples were written. Check the archive path, cutoff, "
                "and source selection."
            )
    finally:
        parser_logger.setLevel(old_parser_level)

    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process raw PDB archive data for DISCO training."
    )
    parser.add_argument("--archive-root", default="/mnt/archive/datasets/disco_training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument(
        "--source",
        choices=["biounit", "mmcif", "protenix"],
        default="biounit",
    )
    parser.add_argument("--cutoff-date", default="2021-09-30")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument(
        "--crop-mode",
        choices=["weighted", "contiguous", "spatial", "interface"],
        default="weighted",
    )
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--samples-per-entry", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cluster-file", default=None)
    parser.add_argument("--seqres-path", default=None)
    parser.add_argument(
        "--protenix-index",
        default=None,
        help=(
            "Optional Protenix index CSV/CSV.GZ. When provided, rows from this "
            "index define the train split, chain/interface samples, and "
            "cluster-size weights."
        ),
    )
    parser.add_argument("--show-parser-warnings", action="store_true")
    parser.add_argument("--all-atom-distogram", action="store_true")
    args = parser.parse_args()

    manifest = process_pdb_archive(
        archive_root=args.archive_root,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        metadata_path=args.metadata,
        source=args.source,
        cutoff_date=args.cutoff_date,
        crop_size=args.crop_size,
        crop_mode=args.crop_mode,
        max_entries=args.max_entries,
        samples_per_entry=args.samples_per_entry,
        seed=args.seed,
        bb_only=not args.all_atom_distogram,
        cluster_file=args.cluster_file,
        seqres_path=args.seqres_path,
        protenix_index=args.protenix_index,
        suppress_parser_warnings=not args.show_parser_warnings,
    )
    print(manifest)


if __name__ == "__main__":
    main()
