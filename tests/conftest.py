from pathlib import Path

import pytest

from benchforge.core.config import RunConfig, load_run_config


@pytest.fixture
def example_config() -> RunConfig:
    return load_run_config(Path("configs/examples/breast_cancer_logreg.yaml"))
