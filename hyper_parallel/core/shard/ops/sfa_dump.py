# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Opt-in local folded sparse-attention snapshots for backward replay."""
import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from hyper_parallel.core.dtensor.dtensor import SkipDTensorDispatch
from hyper_parallel.platform import get_platform
from hyper_parallel.platform.platform import PlatformType

platform = get_platform()
logger = logging.getLogger(__name__)
_LOCK = RLock()
_STATE = None


class _DumpState:
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as source:
            config = json.load(source)
        if not isinstance(config, dict):
            raise ValueError("HP_SFA_DUMP_CONFIG must contain a JSON object.")
        allowed = {"dump_path", "ranks", "steps", "kv_length", "max_samples", "call_indices"}
        if set(config) - allowed:
            raise ValueError(f"Unknown SFA dump options: {sorted(set(config) - allowed)}")
        for name in ("ranks", "steps", "call_indices"):
            values = config.get(name, [] if name == "call_indices" else [0])
            if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
                raise ValueError(f"SFA dump {name} must be a list of nonnegative integers.")
            setattr(self, name, set(values))
        for name, default in (("kv_length", 262144), ("max_samples", 3)):
            value = config.get(name, default)
            if type(value) is not int or value <= 0:
                raise ValueError(f"SFA dump {name} must be a positive integer.")
            setattr(self, name, value)
        dump_path = config.get("dump_path")
        if not isinstance(dump_path, str) or not Path(dump_path).is_absolute():
            raise ValueError("SFA dump dump_path must be an absolute path.")
        if platform.platform_type != PlatformType.MINDSPORE:
            raise ValueError("SFA dump currently supports the MindSpore backend only.")
        self.rank = platform.get_rank()
        self.root = Path(dump_path) / f"rank{self.rank}" / f"run_{uuid4().hex}"
        self.step = None
        self.call_index = 0
        self.samples = []
        self.training_deterministic = None

    def set_step(self, step: Optional[int]) -> None:
        if self.samples:
            completed = sum(sample["status"] == "complete" for sample in self.samples)
            logger.warning("SFA dump rank=%s step=%s: %s/%s complete samples in %s",
                           self.rank, self.step, completed, len(self.samples), self.root)
        elif self.step in self.steps and self.rank in self.ranks:
            logger.warning("SFA dump rank=%s step=%s: no samples; matching grad-enabled calls=%s",
                           self.rank, self.step, self.call_index)
        self.step = step
        self.call_index = 0
        self.samples = []


def set_sfa_dump_step(step: Optional[int], deterministic: Optional[bool] = None) -> None:
    """Open a zero-based optimizer-step window or close it with None.

    Args:
        step: Current optimizer step; disabled without HP_SFA_DUMP_CONFIG.
        deterministic: Deterministic setting declared by the training configuration.
    """
    global _STATE
    config_path = os.environ.get("HP_SFA_DUMP_CONFIG")
    if not config_path:
        return
    with _LOCK:
        if _STATE is None:
            _STATE = _DumpState(config_path)
        _STATE.set_step(step)
        _STATE.training_deterministic = deterministic


def _write_manifest(directory: Path, manifest: dict) -> None:
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def _save_value(directory: Path, name: str, value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_save_value(directory, f"{name}_{index}", item) for index, item in enumerate(value)]
    if not isinstance(value, platform.Tensor):
        raise ValueError(f"Unsupported SFA dump argument {name}: {type(value)}")
    with SkipDTensorDispatch(), platform.no_grad():
        array = np.asarray(platform.tensor_to_numpy(value.detach()))
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
    if array.dtype.itemsize != platform.get_element_size(value) or tuple(array.shape) != tuple(value.shape):
        raise ValueError(f"SFA dump host conversion changed the storage size or shape of {name}.")
    filename = f"{name}.bin"
    array.tofile(directory / filename)
    return {
        "file": filename, "shape": list(array.shape), "dtype": str(value.dtype),
        "numpy_dtype": str(array.dtype), "itemsize": array.dtype.itemsize, "nbytes": array.nbytes,
        "byteorder": "little" if np.little_endian else "big", "storage": "C-order raw bytes",
    }


def dump_sfa_call(inputs: tuple, kwargs: dict, outputs: Any, block_id: int,
                  seq_shard_id: int, seq_shards: int) -> None:
    """Save selected post-fold inputs/outputs and attach an identity gradient hook.

    Args:
        inputs: Actual positional arguments after folding.
        kwargs: Actual keyword arguments after folding.
        outputs: Unstitched attention output, softmax_max and softmax_sum.
        block_id: Folded block identifier.
        seq_shard_id: Sequence shard identifier.
        seq_shards: Number of sequence shards.

    Note:
        Selected tensors are copied synchronously for collection, not timing.
        No collective or explicit device-wide synchronize is added.
    """
    state = _STATE
    if state is None or state.step not in state.steps or state.rank not in state.ranks:
        return
    seq_dim = 0 if kwargs.get("layout_query", "BSND") == "TND" else 1
    if inputs[1].shape[seq_dim] != state.kv_length:
        return
    output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    # Match MindSpore FSDP's flag check to exclude no-grad forward passes.
    if not getattr(output, "_requires_grad", False):
        return
    with _LOCK:
        call_index = state.call_index
        state.call_index += 1
        if len(state.samples) >= state.max_samples or (state.call_indices and call_index not in state.call_indices):
            return
        if len(inputs) != 5 or not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
            raise ValueError("SFA dump requires five positional inputs and three forward outputs.")
        directory = state.root / f"step{state.step}" / f"call{call_index}_block{block_id}"
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": 1, "status": "awaiting_dout", "rank": state.rank,
            "step": state.step, "call_index": call_index, "block_id": block_id,
            "seq_shard_id": seq_shard_id, "seq_shards": seq_shards,
            "training_deterministic": state.training_deterministic,
            "boundary": "post-fold sparse_flash_attention, before output concatenation",
            "inputs": {}, "kwargs": {}, "outputs": {},
            "environment": {key: os.environ.get(key) for key in (
                "MS_DEV_RUNTIME_CONF", "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH")},
        }
        state.samples.append(manifest)
    # Device-to-host copies must not hold the process-wide state lock.
    for name, value in zip(("query", "key", "value", "sparse_indices", "scale_value"), inputs):
        manifest["inputs"][name] = _save_value(directory, name, value)
    for name, value in kwargs.items():
        manifest["kwargs"][name] = _save_value(directory, name, value)
    for name, value in zip(("out", "softmax_max", "softmax_sum"), outputs):
        manifest["outputs"][name] = _save_value(directory, name, value)
    _write_manifest(directory, manifest)

    def save_dout(gradient):
        with _LOCK:
            if manifest["status"] != "awaiting_dout":
                raise RuntimeError(f"SFA dump received multiple gradients for {directory}")
            manifest["status"] = "writing_dout"
        manifest["dout"] = _save_value(directory, "dout", gradient)
        manifest["status"] = "complete"
        _write_manifest(directory, manifest)
        return gradient

    with SkipDTensorDispatch():
        output.register_hook(save_dout)
    logger.warning("SFA dump saved forward: %s; awaiting matching dOut", directory)
