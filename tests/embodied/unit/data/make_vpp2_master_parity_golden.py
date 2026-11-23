# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Write ``vpp2_master_parity_<commit>.json``: master's VPP2 action-stage outputs.

Lane C v2 §3: reference outputs come from the old code and are committed. The
oracle is the one in ``test_arch_vpp2_master_parity.py`` (master's
``scheduler.py``, the ``LiberoActionModel`` training methods and the ``VPP2``
helpers they call, run unmodified), on the inputs that test builds. Only its
CPU cases are stored; the CUDA cases and the heads stay live tests.

Inputs come from fixed seeds and are not stored, so the file also pins torch's
CPU generator: regenerate it when a torch upgrade changes ``torch.randn``, and
point ``REFERENCE`` in the test at the new file. JSON rather than ``.pt``:
``.gitignore`` ignores ``*.pt``.

Usage: python make_vpp2_master_parity_golden.py [/path/to/LoongForge_master_vpp2]
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "vpp2_master_parity", HERE.parent / "test_arch_vpp2_master_parity.py")
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)

# The master files the stored outputs come from.
SOURCES = ("scheduler.py", "libero_action_model.py", "vpp2.py")


def pack(tensor: torch.Tensor) -> dict:
    """Tensor -> JSON; bf16 and float32 values are exact in float64."""
    return {"dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape),
            "data": tensor.detach().cpu().flatten().tolist()}


def collect(master) -> dict[str, torch.Tensor]:
    cpu = torch.device("cpu")
    tensors: dict[str, torch.Tensor] = {}
    for case in parity.CASES:
        result = parity.run_master(master, case, cpu)
        noise = result.pop("video noise")
        key = f"l3/video noise/{case[0]}"
        if key in tensors:
            assert torch.equal(tensors[key], noise), f"{key} differs between cases"
        tensors[key] = noise
        for label, value in result.items():
            tensors[f"l3/{parity.case_id(case)}/{label}"] = value
    for shift in parity.BSMNTW_SHIFTS:
        for name, dtype in parity.DTYPES.items():
            t = parity.bsmntw_inputs(dtype)
            scheduler = master.Scheduler(shift=shift)
            prefix = f"bsmntw/{shift}/{name}/"
            tensors[prefix + "t"] = t
            tensors[prefix + "weight"] = scheduler.training_weight(t)
            tensors[prefix + "weight scalar"] = scheduler.training_weight(t[3:4])
    return tensors


def main(root: Path) -> None:
    legacy = root / "loongforge/embodied/model/vpp2/modules"
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    with pytest.MonkeyPatch.context() as patch:
        tensors = collect(parity.load_master(patch, legacy))
    meta = {"master_commit": commit, "torch": torch.__version__,
            "generator": Path(__file__).name,
            "sources_sha256": {name: hashlib.sha256((legacy / name).read_bytes()).hexdigest()
                               for name in SOURCES}}
    # One tensor per line keeps the file small and its diffs readable.
    body = ",\n".join(f"  {json.dumps(key)}: {json.dumps(pack(value))}"
                      for key, value in tensors.items())
    out = HERE / f"vpp2_master_parity_{commit}.json"
    out.write_text(f'{{\n "meta": {json.dumps(meta)},\n "tensors": {{\n{body}\n }}\n}}\n',
                   encoding="utf-8")
    print(f"wrote {out} ({len(tensors)} tensors)")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]) if len(sys.argv) == 2 else parity.MASTER)
