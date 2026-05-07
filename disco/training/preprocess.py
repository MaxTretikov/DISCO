# Copyright 2026 Jarrid Rector-Brooks, Marta Skreta, Chenghao Liu, Xi Zhang, and Alexander Tong
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable

import biotite.structure as struc
import numpy as np
import torch
from biotite.structure import AtomArray, AtomArrayStack, BondList
from biotite.structure.io import load_structure

from disco.data.ccd import get_ccd_ref_info, get_mol_type
from disco.data.constants import (
    MASK_STD_RESIDUES,
    PRO_STD_RESIDUES,
    PRO_STD_RESIDUES_VALS_SET,
)
from disco.data.featurizer import Featurizer
from disco.data.parser import AddAtomArrayAnnot
from disco.data.tokenizer import AtomArrayTokenizer
from disco.data.utils import data_type_transform, make_dummy_feature


_POLY_TYPES = {
    "protein": "polypeptide(L)",
    "dna": "polydeoxyribonucleotide",
    "rna": "polyribonucleotide",
}


def _first_model(atom_array: AtomArray | AtomArrayStack) -> AtomArray:
    return atom_array[0] if isinstance(atom_array, AtomArrayStack) else atom_array


def _load_atom_array(path: Path) -> AtomArray:
    try:
        atom_array = load_structure(str(path), extra_fields=["occupancy", "b_factor"])
    except TypeError:
        atom_array = load_structure(str(path))
    return _first_model(atom_array)


def _set_required_structure_annotations(atom_array: AtomArray) -> AtomArray:
    if not hasattr(atom_array, "chain_id"):
        atom_array.set_annotation("chain_id", ["A"] * len(atom_array))

    chain_ids = atom_array.chain_id.astype(str)
    chain_to_entity = {
        chain_id: str(i + 1)
        for i, chain_id in enumerate(dict.fromkeys(chain_ids))
    }
    entity_ids = [chain_to_entity[chain_id] for chain_id in chain_ids]

    atom_array.set_annotation("label_entity_id", entity_ids)
    atom_array.set_annotation("label_asym_id", chain_ids)
    atom_array.set_annotation("auth_asym_id", chain_ids)
    atom_array.set_annotation("label_seq_id", atom_array.res_id)
    atom_array.set_annotation("copy_id", [1] * len(atom_array))
    atom_array.set_annotation("is_resolved", [True] * len(atom_array))

    return atom_array


def _ensure_bonds(atom_array: AtomArray) -> AtomArray:
    """Adds standard residue bonds when the source file did not provide them."""
    has_bonds = False
    if atom_array.bonds is not None:
        try:
            has_bonds = len(atom_array.bonds.as_array()) > 0
        except Exception:
            has_bonds = True

    if not has_bonds:
        try:
            atom_array.bonds = struc.connect_via_residue_names(atom_array)
        except Exception:
            atom_array.bonds = BondList(len(atom_array))
    return atom_array


def _infer_entity_poly_type(atom_array: AtomArray) -> dict[str, str]:
    entity_to_mol_types: dict[str, set[str]] = {}
    for residue in struc.residue_iter(atom_array):
        entity_id = str(residue.label_entity_id[0])
        try:
            mol_type = get_mol_type(residue.res_name[0])
        except Exception:
            mol_type = "ligand"
        entity_to_mol_types.setdefault(entity_id, set()).add(mol_type)

    entity_poly_type = {}
    for entity_id, mol_types in entity_to_mol_types.items():
        for mol_type in ("protein", "dna", "rna"):
            if mol_type in mol_types:
                entity_poly_type[entity_id] = _POLY_TYPES[mol_type]
                break
    return entity_poly_type


def _filter_supported_atoms(atom_array: AtomArray) -> AtomArray:
    ref_cache = {}
    keep = []
    for atom in atom_array:
        if atom.element == "H" or atom.res_name in {"HOH", "WAT"}:
            keep.append(False)
            continue

        ref_info = ref_cache.get(atom.res_name)
        if atom.res_name not in ref_cache:
            try:
                ref_info = get_ccd_ref_info(atom.res_name, return_perm=False)
            except Exception:
                ref_info = None
            ref_cache[atom.res_name] = ref_info

        if ref_info and atom.atom_name not in ref_info["atom_map"]:
            keep.append(False)
        else:
            keep.append(True)

    return atom_array[keep]


def prepare_training_atom_array(
    path: str | Path,
    bb_only: bool = True,
) -> tuple[AtomArray, dict[str, str]]:
    """Loads a structure file and adds DISCO feature annotations."""
    atom_array = _load_atom_array(Path(path))
    atom_array = _set_required_structure_annotations(atom_array)
    atom_array = _filter_supported_atoms(atom_array)
    atom_array = _ensure_bonds(atom_array)
    entity_poly_type = _infer_entity_poly_type(atom_array)

    atom_array = AddAtomArrayAnnot.add_token_mol_type(atom_array, entity_poly_type)
    atom_array = AddAtomArrayAnnot.add_centre_atom_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_atom_mol_type_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_distogram_rep_atom_mask(atom_array, bb_only=bb_only)
    atom_array = AddAtomArrayAnnot.add_plddt_m_rep_atom_mask(atom_array)
    atom_array = AddAtomArrayAnnot.add_cano_seq_resname(atom_array)
    atom_array = AddAtomArrayAnnot.add_tokatom_idx(atom_array)
    atom_array = AddAtomArrayAnnot.add_modified_res_mask(atom_array)
    atom_array = AddAtomArrayAnnot.unique_chain_and_add_ids(atom_array)
    atom_array = AddAtomArrayAnnot.find_equiv_mol_and_assign_ids(atom_array)
    atom_array = AddAtomArrayAnnot.add_ref_space_uid(atom_array)
    atom_array = AddAtomArrayAnnot.add_ref_info_and_res_perm(atom_array)
    return atom_array, entity_poly_type


def crop_atom_array_by_tokens(
    atom_array: AtomArray,
    max_tokens: int | None = 384,
    *,
    seed: int | None = None,
    start: int | None = None,
) -> AtomArray:
    """Crops an annotated structure to a contiguous token window.

    Standard polymer residues are kept whole because they are single DISCO
    tokens. Ligands and non-standard residues are atom-level tokens, matching
    the tokenizer used by the model features.
    """
    if max_tokens is None:
        return atom_array
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive or None.")

    token_array = AtomArrayTokenizer(atom_array).get_token_array()
    if len(token_array) <= max_tokens:
        return atom_array

    max_start = len(token_array) - max_tokens
    if start is None:
        if seed is None:
            start = 0
        else:
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, max_start + 1))

    if start < 0 or start > max_start:
        raise ValueError(
            f"Crop start {start} is invalid for {len(token_array)} tokens "
            f"and crop size {max_tokens}."
        )

    selected_tokens = token_array[start : start + max_tokens]
    atom_indices = [
        atom_index for token in selected_tokens for atom_index in token.atom_indices
    ]
    return atom_array[atom_indices]


def atom_array_to_training_example(atom_array: AtomArray) -> dict:
    """Converts an annotated AtomArray to one manifest-backed training example."""
    token_array = AtomArrayTokenizer(atom_array).get_token_array()
    feature_dict = Featurizer(token_array, atom_array).get_all_input_features()
    feature_dict = make_dummy_feature(feature_dict, dummy_feats=["msa", "template"])
    feature_dict = data_type_transform(feature_dict)

    feature_dict["prot_residue_mask"] = torch.tensor(
        [token.value in PRO_STD_RESIDUES_VALS_SET for token in token_array],
        dtype=torch.bool,
    )

    valid_residues = PRO_STD_RESIDUES | MASK_STD_RESIDUES
    true_prot_restype = torch.tensor(
        [
            token.value
            for token in token_array
            if token.value in PRO_STD_RESIDUES_VALS_SET
            and token.value in valid_residues.values()
        ],
        dtype=torch.long,
    )
    if true_prot_restype.numel() == 0:
        raise ValueError("No protein residues found in structure.")

    coordinate = torch.as_tensor(atom_array.coord, dtype=torch.float32)
    if hasattr(atom_array, "is_resolved"):
        coordinate_mask = torch.as_tensor(atom_array.is_resolved, dtype=torch.bool)
    else:
        coordinate_mask = torch.ones(len(atom_array), dtype=torch.bool)

    return {
        "input_feature_dict": feature_dict,
        "coordinate": coordinate,
        "coordinate_mask": coordinate_mask,
        "true_prot_restype": true_prot_restype,
        "distogram_rep_atom_mask": feature_dict["distogram_rep_atom_mask"],
    }


def preprocess_structure_file(
    input_path: str | Path,
    output_path: str | Path,
    bb_only: bool = True,
    crop_size: int | None = 384,
    crop_seed: int | None = None,
    crop_start: int | None = None,
) -> Path:
    """Writes one preprocessed `.pt` training example."""
    atom_array, _ = prepare_training_atom_array(input_path, bb_only=bb_only)
    atom_array = crop_atom_array_by_tokens(
        atom_array,
        max_tokens=crop_size,
        seed=crop_seed,
        start=crop_start,
    )
    example = atom_array_to_training_example(atom_array)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(example, output_path)
    return output_path


def _expand_inputs(inputs: Iterable[str], input_globs: Iterable[str]) -> list[Path]:
    paths = [Path(path) for path in inputs]
    for pattern in input_globs:
        paths.extend(Path(path) for path in glob.glob(pattern))
    return sorted(dict.fromkeys(paths))


def preprocess_to_manifest(
    inputs: Iterable[str],
    input_globs: Iterable[str],
    output_dir: str | Path,
    manifest_path: str | Path,
    bb_only: bool = True,
    crop_size: int | None = 384,
    crop_seed: int | None = None,
) -> Path:
    """Preprocesses structures and writes a relative-path manifest."""
    input_paths = _expand_inputs(inputs, input_globs)
    if len(input_paths) == 0:
        raise ValueError("No input structure files were provided.")

    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for index, input_path in enumerate(input_paths):
        output_path = output_dir / f"{input_path.stem}.pt"
        per_file_seed = None if crop_seed is None else crop_seed + index
        output_paths.append(
            preprocess_structure_file(
                input_path,
                output_path,
                bb_only=bb_only,
                crop_size=crop_size,
                crop_seed=per_file_seed,
            )
        )

    manifest_lines = [
        str(
            path.relative_to(manifest_path.parent)
            if path.is_relative_to(manifest_path.parent)
            else path
        )
        for path in output_paths
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess structure files for DISCO training.")
    parser.add_argument("--input", nargs="*", default=[], help="Input PDB/mmCIF files.")
    parser.add_argument("--input-glob", nargs="*", default=[], help="Glob(s) for input files.")
    parser.add_argument("--output-dir", required=True, help="Directory for `.pt` examples.")
    parser.add_argument("--manifest", required=True, help="Output manifest path.")
    parser.add_argument(
        "--all-atom-distogram",
        action="store_true",
        help="Use non-backbone distogram representatives where available.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=384,
        help="Maximum tokens per example. Use 0 to disable cropping.",
    )
    parser.add_argument(
        "--crop-seed",
        type=int,
        default=None,
        help="Seed for random contiguous crops. Defaults to the first crop window.",
    )
    args = parser.parse_args()

    crop_size = None if args.crop_size <= 0 else args.crop_size
    manifest = preprocess_to_manifest(
        inputs=args.input,
        input_globs=args.input_glob,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        bb_only=not args.all_atom_distogram,
        crop_size=crop_size,
        crop_seed=args.crop_seed,
    )
    print(manifest)


if __name__ == "__main__":
    main()
