"""Shared fixtures. All fixtures are offline and deterministic."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore", category=DeprecationWarning)

from epitope_map import demo as synthetic  # noqa: E402

from epitope_map.pipeline import RunConfig, run_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_inputs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("synthetic")
    return synthetic.write_inputs(directory)


@pytest.fixture(scope="session")
def synthetic_config(synthetic_inputs, tmp_path_factory):
    outdir = tmp_path_factory.mktemp("results")
    return RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(synthetic_inputs["binding"]),
        reference="mouse",
        structure=str(synthetic_inputs["structure"]),
        outdir=outdir,
    )


@pytest.fixture(scope="session")
def synthetic_result(synthetic_config):
    return run_pipeline(synthetic_config)
