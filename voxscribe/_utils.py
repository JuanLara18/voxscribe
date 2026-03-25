"""Shared internal utilities for VoxScribe.

Not part of the public API — subject to change without notice.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on availability.

    Args:
        requested: One of 'auto', 'cuda', 'cpu'.

    Returns:
        Resolved device string ('cuda' or 'cpu').
    """
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            logger.debug("CUDA available — using GPU.")
            return "cuda"
    except ImportError:
        pass
    logger.debug("CUDA not available — falling back to CPU.")
    return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Resolve 'auto' compute type based on device.

    For CUDA devices, float16 provides best speed/quality tradeoff.
    For CPU, int8 is significantly faster than float32 with minimal quality loss.

    Args:
        compute_type: One of 'auto', 'float16', 'int8', 'float32'.
        device: Resolved device string.

    Returns:
        Resolved compute type string.
    """
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def format_timestamp(seconds: float, ms_separator: str = ",") -> str:
    """Format seconds as SRT/VTT timestamp (HH:MM:SS,mmm or HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds (float).
        ms_separator: ',' for SRT, '.' for VTT.

    Returns:
        Formatted timestamp string.
    """
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}{ms_separator}{ms:03d}"


def format_timestamp_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS (no milliseconds).

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted timestamp string.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
