"""
Bootstrap utility for path setup and project root detection.

This module handles all the messy path manipulation in one place,
so other files can simply import this and get clean access to utilities.
"""

import sys
from pathlib import Path


def setup_project_paths():
    """
    Set up Python paths and return project root.
    Call this once at the top of any module that needs project utilities.
    """
    # Find project root by looking for marker files
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in ['.git', 'requirements.txt', 'README.md']):
            project_root = parent
            break
    else:
        raise RuntimeError("Could not find project root")
    
    # Add src directory to Python path if not already there
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    return project_root


# Auto-setup when imported
PROJECT_ROOT = setup_project_paths()


def get_project_root():
    """Get the project root directory."""
    return PROJECT_ROOT


def get_project_outputs_dir():
    """Get the project outputs directory."""
    return PROJECT_ROOT / "outputs"


def get_project_data_dir():
    """Get the project data directory.""" 
    return PROJECT_ROOT / "data"


def get_project_logs_dir():
    """Get the project logs directory."""
    return PROJECT_ROOT / "logs"