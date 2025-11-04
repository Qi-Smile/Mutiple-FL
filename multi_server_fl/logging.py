from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

try:
    import swanlab

    _SWANLAB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    swanlab = None
    _SWANLAB_AVAILABLE = False


@dataclass
class LoggerConfig:
    project: str = "multi-server-fl"
    run_name: Optional[str] = None
    enable_swanlab: bool = False


class MetricLogger:
    """Unified interface for optional SwanLab logging."""

    def __init__(self, config: LoggerConfig | None = None) -> None:
        self.config = config or LoggerConfig()
        self._swanlab_run = None
        if self.config.enable_swanlab:
            if not _SWANLAB_AVAILABLE:
                raise RuntimeError("SwanLab requested but not installed.")
            self._swanlab_run = swanlab.init(
                project=self.config.project,
                run=self.config.run_name,
            )

    def log(self, metrics: Dict[str, float], step: int | None = None) -> None:
        if self._swanlab_run is not None:
            self._swanlab_run.log(metrics, step=step)

    def finish(self) -> None:
        if self._swanlab_run is not None:
            self._swanlab_run.finish()
