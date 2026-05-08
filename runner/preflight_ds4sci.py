#!/usr/bin/env python
"""Preflight checks for DeepSpeed4Science EvoformerAttention.

DeepSpeed's Evoformer op builder only reports a boolean compatibility result.
This script does the environment work around that check: find plausible CUTLASS
roots, validate their shape, and show the exact variables to export before
training or inference.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MIN_CUTLASS_VERSION = (3, 1, 0)
SPECIAL_CUTLASS_VALUES = {
    "DS_IGNORE_CUTLASS_DETECTION",
    "DS_USE_CUTLASS_PYTHON_BINDINGS",
}


@dataclass
class CutlassCandidate:
    value: str
    source: str
    exists: bool
    include_dir: str | None
    util_include_dir: str | None
    has_cutlass_headers: bool
    has_cute_headers: bool
    version: str | None
    version_ok: bool | None
    valid: bool
    reason: str


@dataclass
class BuilderCheck:
    compatible: bool
    include_paths: list[str]
    error: str | None


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command,
            cwd=cwd,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parse_version(text: str) -> str | None:
    for pattern in [
        r"^## \[?v?(\d+\.\d+\.\d+)",
        r"^# \[?v?(\d+\.\d+\.\d+)",
        r"v?(\d+\.\d+\.\d+)",
    ]:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1)
    return None


def _version_tuple(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")[:3]
    return int(major), int(minor), int(patch)


def _cutlass_version(path: Path) -> str | None:
    changelog = path / "CHANGELOG.md"
    if changelog.is_file():
        version = _parse_version(changelog.read_text(errors="replace"))
        if version:
            return version

    version_header = path / "include" / "cutlass" / "version.h"
    if version_header.is_file():
        return _parse_version(version_header.read_text(errors="replace"))

    git_version = _run(["git", "describe", "--tags", "--abbrev=0"], cwd=path)
    if git_version:
        return _parse_version(git_version)

    return None


def _candidate_from_value(value: str, source: str) -> CutlassCandidate:
    if value in SPECIAL_CUTLASS_VALUES:
        return CutlassCandidate(
            value=value,
            source=source,
            exists=True,
            include_dir=None,
            util_include_dir=None,
            has_cutlass_headers=False,
            has_cute_headers=False,
            version=None,
            version_ok=None,
            valid=True,
            reason="DeepSpeed special CUTLASS_PATH value; no local path validation performed.",
        )

    path = Path(value).expanduser()
    exists = path.is_dir()
    include_dir = path / "include"
    util_include_dir = path / "tools" / "util" / "include"
    has_cutlass_headers = (include_dir / "cutlass").is_dir()
    has_cute_headers = (include_dir / "cute").is_dir()
    version = _cutlass_version(path) if exists else None
    version_ok = None if version is None else _version_tuple(version) >= MIN_CUTLASS_VERSION

    valid = exists and has_cutlass_headers and version_ok is not False
    if not exists:
        reason = "Directory does not exist."
    elif not include_dir.is_dir():
        reason = "Missing include/ directory."
    elif not has_cutlass_headers:
        reason = "Missing include/cutlass headers."
    elif version_ok is False:
        reason = f"CUTLASS {version} is older than 3.1.0."
    elif not util_include_dir.is_dir():
        reason = "Valid, but tools/util/include is missing; JIT may still compile without it."
    else:
        reason = "Valid CUTLASS checkout."

    return CutlassCandidate(
        value=str(path),
        source=source,
        exists=exists,
        include_dir=str(include_dir) if include_dir.is_dir() else None,
        util_include_dir=str(util_include_dir) if util_include_dir.is_dir() else None,
        has_cutlass_headers=has_cutlass_headers,
        has_cute_headers=has_cute_headers,
        version=version,
        version_ok=version_ok,
        valid=valid,
        reason=reason,
    )


def _add_candidate(values: list[tuple[str, str]], seen: set[str], value: str, source: str) -> None:
    key = value if value in SPECIAL_CUTLASS_VALUES else str(Path(value).expanduser())
    if key not in seen:
        seen.add(key)
        values.append((value, source))


def discover_cutlass(explicit: str | None = None) -> list[CutlassCandidate]:
    values: list[tuple[str, str]] = []
    seen: set[str] = set()

    if explicit:
        _add_candidate(values, seen, explicit, "--cutlass-path")

    env_value = os.environ.get("CUTLASS_PATH")
    if env_value:
        _add_candidate(values, seen, env_value, "CUTLASS_PATH")

    for pattern in [
        "/usr/local/src/cutlass*",
        "/usr/local/cutlass*",
        "/opt/cutlass*",
        "/mnt/archive/cutlass*",
        "/mnt/archive/datasets/cutlass*",
        "/mnt/shared/cutlass*",
        str(Path.cwd() / "cutlass*"),
    ]:
        for path in sorted(Path("/").glob(pattern.lstrip("/"))):
            if path.is_dir():
                _add_candidate(values, seen, str(path), f"glob:{pattern}")

    if Path("/usr/local/include/cutlass").is_dir():
        _add_candidate(values, seen, "/usr/local", "system include")

    return [_candidate_from_value(value, source) for value, source in values]


@contextlib.contextmanager
def _temporary_env(key: str, value: str):
    previous = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def check_deepspeed_builder(cutlass_value: str) -> BuilderCheck:
    try:
        with _temporary_env("CUTLASS_PATH", cutlass_value):
            from deepspeed.ops.op_builder.evoformer_attn import EvoformerAttnBuilder

            builder = EvoformerAttnBuilder()
            compatible = bool(builder.is_compatible(verbose=True))
            try:
                include_paths = [str(path) for path in builder.include_paths()]
            except Exception:
                include_paths = []
        return BuilderCheck(compatible=compatible, include_paths=include_paths, error=None)
    except Exception as exc:
        return BuilderCheck(
            compatible=False,
            include_paths=[],
            error=f"{type(exc).__name__}: {exc}",
        )


def system_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cuda_path": os.environ.get("CUDA_PATH"),
        "cutlass_path": os.environ.get("CUTLASS_PATH"),
        "nvcc": shutil.which("nvcc"),
        "nvcc_version": None,
        "torch": None,
        "torch_cuda": None,
        "cuda_available": None,
        "gpu": None,
        "capability": None,
        "deepspeed": None,
    }
    if report["nvcc"]:
        nvcc_output = _run([str(report["nvcc"]), "--version"])
        if nvcc_output:
            report["nvcc_version"] = nvcc_output.splitlines()[-1]

    try:
        import torch

        report["torch"] = torch.__version__
        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            report["capability"] = ".".join(str(part) for part in capability)
    except Exception as exc:
        report["torch_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import deepspeed

        report["deepspeed"] = deepspeed.__version__
    except Exception as exc:
        report["deepspeed_error"] = f"{type(exc).__name__}: {exc}"

    return report


def choose_candidate(candidates: list[CutlassCandidate]) -> CutlassCandidate | None:
    for source in ["--cutlass-path", "CUTLASS_PATH"]:
        for candidate in candidates:
            if candidate.source == source and candidate.valid:
                return candidate
    return next((candidate for candidate in candidates if candidate.valid), None)


def print_human(
    report: dict[str, Any],
    candidates: list[CutlassCandidate],
    chosen: CutlassCandidate | None,
    builder_check: BuilderCheck | None,
) -> None:
    print("DeepSpeed4Science EvoformerAttention preflight")
    print("=" * 54)
    print(f"Python:       {report.get('python')} ({report.get('executable')})")
    print(f"PyTorch:      {report.get('torch')} CUDA {report.get('torch_cuda')}")
    print(f"DeepSpeed:    {report.get('deepspeed')}")
    print(f"CUDA_PATH:    {report.get('cuda_path')}")
    print(f"nvcc:         {report.get('nvcc')}")
    print(f"nvcc version: {report.get('nvcc_version')}")
    print(f"GPU:          {report.get('gpu')} sm_{str(report.get('capability')).replace('.', '')}")
    print()

    if not candidates:
        print("CUTLASS candidates: none found")
    else:
        print("CUTLASS candidates:")
        for candidate in candidates:
            status = "OK" if candidate.valid else "BAD"
            version = candidate.version or "unknown"
            print(f"- [{status}] {candidate.value} ({candidate.source}, version={version})")
            print(f"  {candidate.reason}")
            if candidate.include_dir:
                print(f"  include: {candidate.include_dir}")
            if candidate.util_include_dir:
                print(f"  util:    {candidate.util_include_dir}")
    print()

    if chosen is None:
        print("No usable CUTLASS path found.")
        print("Install CUTLASS or pass --cutlass-path /path/to/cutlass.")
        return

    print(f"Selected CUTLASS_PATH: {chosen.value}")
    if builder_check is not None:
        status = "OK" if builder_check.compatible else "BAD"
        print(f"DeepSpeed Evoformer builder compatibility: {status}")
        if builder_check.include_paths:
            print("Builder include paths:")
            for path in builder_check.include_paths:
                print(f"- {path}")
        if builder_check.error:
            print(f"Builder error: {builder_check.error}")
    print()
    print("Use this before training/inference:")
    print(f"export CUTLASS_PATH={chosen.value}")
    if report.get("capability"):
        print(f"export TORCH_CUDA_ARCH_LIST={report['capability']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and validate CUTLASS for DeepSpeed4Science EvoformerAttention.",
    )
    parser.add_argument("--cutlass-path", help="Explicit CUTLASS checkout/root to validate.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--skip-builder-check",
        action="store_true",
        help="Skip importing DeepSpeed's Evoformer op builder.",
    )
    args = parser.parse_args()

    report = system_report()
    candidates = discover_cutlass(args.cutlass_path)
    chosen = choose_candidate(candidates)
    builder_check = None
    if chosen is not None and not args.skip_builder_check:
        builder_check = check_deepspeed_builder(chosen.value)

    if args.json:
        print(
            json.dumps(
                {
                    "system": report,
                    "candidates": [asdict(candidate) for candidate in candidates],
                    "selected": asdict(chosen) if chosen else None,
                    "builder_check": asdict(builder_check) if builder_check else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_human(report, candidates, chosen, builder_check)

    if chosen is None:
        return 2
    if builder_check is not None and not builder_check.compatible:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
