from __future__ import annotations

import logging
import warnings
import sys
import io
from typing import Optional
from rich.logging import RichHandler


class _SuppressConsoleNoise(logging.Filter):
    """Filter out verbose training messages from the console output."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple filter
        msg = record.getMessage()
        if "learn:" in msg:
            return False
        if "The objective has been evaluated at this point before" in msg:
            return False
        return True


class _StreamToLogger(io.TextIOBase):
    """Redirect writes to a logger."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level

    def write(self, message: str) -> int:  # pragma: no cover - thin wrapper
        message = message.strip()
        if message:
            for line in message.splitlines():
                self.logger.log(self.level, line)
        return len(message)

    def flush(self) -> None:  # pragma: no cover - required for file-like API
        pass


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure logging to write to ``log_file`` and keep the console clean."""

    handlers: list[logging.Handler] = []
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(level)
    console_handler.addFilter(_SuppressConsoleNoise())
    handlers.append(console_handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    logging.captureWarnings(True)

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources.*deprecated.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*force_all_finite.*renamed.*",
        category=FutureWarning,
    )

    if log_file is not None:
        stdout_logger = logging.getLogger("STDOUT")
        stderr_logger = logging.getLogger("STDERR")
        stdout_logger.setLevel(logging.INFO)
        stderr_logger.setLevel(logging.ERROR)
        for handler in handlers:
            stdout_logger.addHandler(handler)
            stderr_logger.addHandler(handler)
        sys.stdout = _StreamToLogger(stdout_logger, logging.INFO)
        sys.stderr = _StreamToLogger(stderr_logger, logging.ERROR)

