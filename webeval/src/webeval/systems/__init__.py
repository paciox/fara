"""Systems for task execution."""

from .websurfer import WebSurferSystem, WebSurferSystemOrchestrator
from .computer_use import ComputerUseSystem, ComputerUseOrchestrator

__all__ = [
    "WebSurferSystem",
    "WebSurferSystemOrchestrator",
    "ComputerUseSystem",
    "ComputerUseOrchestrator",
]
