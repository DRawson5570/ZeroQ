"""
Pytest configuration for ZeRO-Q tests.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "distributed: marks tests that require multiple GPUs"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


@pytest.fixture(scope="session")
def gpu_available():
    """Check if GPU is available."""
    import torch
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def bnb_available():
    """Check if bitsandbytes is available."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False
