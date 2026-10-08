"""Minimal safetensors reader (single-file tensors only).

Implements the subset of the safetensors format used by the released
checkpoint (model.safetensors): JSON header + dense little-endian float16/
float32 tensors. No third-party library required.

Format reference: https://huggingface.co/docs/safetensors/
"""

import json
import struct

import torch

_DTYPES = {
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
}


def load_safetensors(path):
    """Load a single-file safetensors checkpoint into {name: torch.Tensor}."""
    with open(path, "rb") as fh:
        data = fh.read()
    if len(data) < 8:
        raise ValueError("safetensors: file too short (%s)" % path)
    header_len = struct.unpack("<Q", data[:8])[0]
    if header_len > 100 * 1024 * 1024:
        raise ValueError("safetensors: header implausibly large (%d)" % header_len)
    header = json.loads(data[8:8 + header_len].decode("utf-8"))
    base = 8 + header_len
    tensors = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype, size = _DTYPES[info["dtype"]]
        shape = info["shape"]
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * size
        off = info["data_offsets"]
        raw = data[base + off[0]:base + off[1]]
        if len(raw) != nbytes:
            raise ValueError("safetensors: truncated tensor %r" % name)
        tensors[name] = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape)
    return tensors
