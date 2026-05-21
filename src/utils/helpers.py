
"""
Utility helpers shared across the project.
"""

import os
import random
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np


# ─────────────────────────────────────────────
#  Project Root Resolution
# ─────────────────────────────────────────────
def get_project_root() -> Path:
    """Return absolute path to the sentiment_analysis project root."""
    return Path(__file__).resolve().parents[2]


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and return the project YAML config."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────
#  Directory Helpers
# ─────────────────────────────────────────────
def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_path(*parts: str) -> Path:
    """Build a path relative to the project root."""
    return get_project_root().joinpath(*parts)


# ─────────────────────────────────────────────
#  Device Detection
# ─────────────────────────────────────────────
def get_device() -> str:
    """Return 'cuda' if GPU is available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def device_info() -> Dict[str, Any]:
    """Return a summary dict about the available compute device."""
    info: Dict[str, Any] = {"device": get_device()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 2
            )
    except ImportError:
        pass
    return info
