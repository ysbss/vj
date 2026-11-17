#!/usr/bin/env python3
"""Detect inference device for Demucs (torch) and faster-whisper (ctranslate2)."""
from __future__ import annotations

import sys


def detect_demucs_device() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return "cuda", name
    except Exception:
        pass
    return "cpu", ""


def detect_whisper_device() -> str:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    demucs_device, gpu_name = detect_demucs_device()
    whisper_device = detect_whisper_device()
    compute_type = "float16" if whisper_device == "cuda" else "int8"
    cuda_available = demucs_device == "cuda" or whisper_device == "cuda"

    print(f"demucs_device={demucs_device}")
    print(f"whisper_device={whisper_device}")
    print(f"compute_type={compute_type}")
    print(f"cuda_available={'true' if cuda_available else 'false'}")
    print(f"gpu_name={gpu_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
