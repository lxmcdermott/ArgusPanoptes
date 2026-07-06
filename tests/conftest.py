"""Pytest configuration: make the repo root importable and share fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensors import SawVibrationSimulator, ThermalSimulator, load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture()
def vib(config):
    return SawVibrationSimulator(config)


@pytest.fixture()
def therm(config):
    return ThermalSimulator(config)
