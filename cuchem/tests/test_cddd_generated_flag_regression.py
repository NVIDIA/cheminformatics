"""Regression test for off-by-one column index in Cddd Generated flag.

The bug wrote ``iat[0, 2] = False`` (and ``iat[-1, 2] = False``) into the
``embeddings_dim`` column instead of the ``Generated`` column, leaving the
source/destination molecules marked as generated and corrupting their
embedding dimensions.

This test loads ``cuchem/cuchem/wf/generative/cddd.py`` in isolation with the
heavy model/embedding dependencies mocked, so it can run with only
``pandas``, ``numpy`` and ``pytest``.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# Fixed 3-D embedding returned by the mocked CDDD encoder.
_FIXED_EMBEDDING = np.array([1.0, 2.0, 3.0])


def _inverse_transform_mock(embeddings):
    """Return deterministic SMILES strings for any embedding input."""
    arr = np.asarray(embeddings)
    if arr.ndim >= 2:
        return np.array([f"SMILES_{i}" for i in range(arr.shape[0])])
    return [f"SMILES_{i}" for i in range(arr.shape[0])]


def _transform_mock(data):
    """Return two fixed embeddings for interpolation inputs."""
    return np.array([_FIXED_EMBEDDING, np.array([4.0, 5.0, 6.0])])


class _MockBaseGenerativeWorkflow:
    """Minimal stand-in for BaseGenerativeWorkflow used by Cddd."""

    def __init__(self, dao=None):
        self.dao = dao
        self.min_jitter_radius = 0.5

    def _compute_radius(self, scaled_radius):
        if scaled_radius:
            return float(scaled_radius * self.min_jitter_radius)
        return self.min_jitter_radius

    def addjitter(self, embedding, radius=None, cnt=1, shape=None):
        radius = radius if radius else self.min_jitter_radius
        rng = np.random.default_rng(42)
        return [embedding + rng.normal(0, radius, embedding.shape) for _ in range(cnt)]

    def compute_unique_smiles(self, interp_df, embedding_funct, scaled_radius=None):
        # Not exercised when force_unique=False; return unchanged.
        return interp_df


@pytest.fixture
def cddd_module():
    """Load cddd.py directly with all heavy dependencies mocked."""
    # Mocks for modules imported by cddd.py.
    sys.modules["cuchem.utils.data_peddler"] = MagicMock(
        download_cddd_models=MagicMock(return_value="/tmp/mock_cddd")
    )

    mock_fingerprint = MagicMock()
    mock_embeddings_cls = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings_cls.return_value = mock_embeddings
    mock_embeddings.func.seq_to_emb.return_value = _FIXED_EMBEDDING
    mock_embeddings.inverse_transform.side_effect = _inverse_transform_mock
    mock_embeddings.transform.side_effect = _transform_mock
    mock_fingerprint.Embeddings = mock_embeddings_cls
    sys.modules["cuchemcommon.fingerprint"] = mock_fingerprint

    sys.modules["cuchemcommon.data"] = MagicMock()
    sys.modules["cuchemcommon.data.generative_wf"] = MagicMock()
    sys.modules["cuchemcommon.utils.singleton"] = MagicMock(
        Singleton=type  # disable singleton behaviour for tests
    )
    sys.modules["cuchemcommon.workflow"] = MagicMock(
        BaseGenerativeWorkflow=_MockBaseGenerativeWorkflow
    )

    # Load cddd.py directly so we do not import the heavy generative package
    # __init__ (which pulls in megatronmolbart/grpc dependencies).
    source_path = (
        Path(__file__).resolve().parents[2]
        / "cuchem"
        / "cuchem"
        / "wf"
        / "generative"
        / "cddd.py"
    )
    spec = importlib.util.spec_from_file_location("cddd_regression", source_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cddd_regression"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cddd_instance(cddd_module):
    return cddd_module.Cddd()


def test_find_similars_smiles_marks_source_not_generated(cddd_instance):
    df = cddd_instance.find_similars_smiles("CCO", num_requested=3)

    # The source SMILES row must be marked as not generated.
    assert df.at[0, "Generated"] == False
    # The off-by-one bug wrote False into the embeddings_dim column instead.
    assert df.at[0, "embeddings_dim"] is not False


def test_interpolate_smiles_marks_endpoints_not_generated(cddd_instance):
    df = cddd_instance.interpolate_smiles(["CCO", "CCCO"], num_points=3)

    last_idx = df.index[-1]

    # Both endpoints are user-supplied input molecules, not generated.
    assert df.at[0, "Generated"] == False
    assert df.at[last_idx, "Generated"] == False

    # embeddings_dim must remain a shape tuple, not be clobbered with False.
    assert df.at[0, "embeddings_dim"] is not False
    assert df.at[last_idx, "embeddings_dim"] is not False
