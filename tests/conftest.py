# conftest.py
import pytest

from typing import Any

def pytest_addoption(parser: Any):
    parser.addoption("--visualize", action="store_true", help="run visualization")

@pytest.fixture
def visualize(request: Any):
    return request.config.getoption("--visualize")