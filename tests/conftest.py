import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPECTRAL_DIR = ROOT / "src" / "sidescatter" / "demo_data" / "spectral"

# FlowKit's repository data (FlowJo 10 workspaces with FCS files). Set
# FLOWKIT_DATA to a clone's data/ folder to run the FlowJo comparison tests.
FLOWKIT_DATA = Path(os.environ.get("FLOWKIT_DATA", ROOT / ".testdata" / "FlowKit" / "data"))


@pytest.fixture
def spectral_dir():
    return SPECTRAL_DIR


@pytest.fixture
def eight_color():
    d = FLOWKIT_DATA / "8_color_data_set"
    if not d.exists():
        pytest.skip("FlowKit test data not available (set FLOWKIT_DATA)")
    return d
