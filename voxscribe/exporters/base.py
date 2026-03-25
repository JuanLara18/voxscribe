"""Abstract protocol for output format exporters."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from voxscribe.models import MergedSegment


@runtime_checkable
class BaseExporter(Protocol):
    """Structural protocol for all output format exporters."""

    def export(
        self,
        segments: list[MergedSegment],
        output_path: Path,
        *,
        title: str | None = None,
        summary: str | None = None,
    ) -> None:
        """Write *segments* to *output_path* in the exporter's format.

        Args:
            segments: Merged transcript segments.
            output_path: Destination file path (parent directory must exist).
            title: Optional document title (used by formats that support it).
            summary: Optional LLM-generated summary (used by Markdown exporter).
        """
        ...
