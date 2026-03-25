"""Output format exporters subpackage."""

from __future__ import annotations

from voxscribe.exporters.base import BaseExporter


def get_exporter(fmt: str) -> BaseExporter:
    """Return the exporter instance for *fmt*.

    Args:
        fmt: Format key — one of ``'md'``, ``'json'``, ``'srt'``, ``'vtt'``, ``'txt'``.

    Returns:
        An object satisfying the :class:`BaseExporter` protocol.

    Raises:
        ValueError: For an unknown format key.
    """
    fmt = fmt.lower().strip(".")

    if fmt == "json":
        from voxscribe.exporters.json_exporter import JSONExporter  # noqa

        return JSONExporter()
    if fmt in ("md", "markdown"):
        from voxscribe.exporters.markdown_exporter import MarkdownExporter  # noqa

        return MarkdownExporter()
    if fmt == "srt":
        from voxscribe.exporters.srt_exporter import SRTExporter  # noqa

        return SRTExporter()
    if fmt == "vtt":
        from voxscribe.exporters.vtt_exporter import VTTExporter  # noqa

        return VTTExporter()
    if fmt == "txt":
        from voxscribe.exporters.txt_exporter import TXTExporter  # noqa

        return TXTExporter()

    raise ValueError(
        f"Unknown output format: '{fmt}'. "
        "Supported: 'md', 'json', 'srt', 'vtt', 'txt'."
    )


__all__ = ["BaseExporter", "get_exporter"]
