from __future__ import annotations

import logging
import warnings
from rich.logging import RichHandler


def setup_logging(level: int = logging.INFO) -> None:
    """Configure Rich logging and silence noisy warnings."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

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

