"""Conformer seeds must follow molecules through parallel and fallback paths."""

import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from monroe.eval.embed import _featurize_one, embed_smiles, featurize_smiles
from monroe.model.ckpt import _build_encoder
from monroe.preprocessing import build_eval_graphs as graphs


# Nine chunks at chunksize=8: workers must take more than one chunk.
SMILES = [
    "C" * length + tail
    for length in range(1, 10)
    for tail in ("O", "N", "F", "Cl", "Br", "C(=O)O", "C(=O)N", "OC")
]


@pytest.mark.parametrize("seed_args", [{}, {"conformer_seed": 123}])
def test_worker_count_repeated_calls_and_order(seed_args):
    reference = dict(featurize_smiles(SMILES, n_workers=1, **seed_args))
    assert len(reference) == len(SMILES)
    for workers, smiles in ((8, SMILES), (8, SMILES), (4, SMILES[::-1])):
        actual = dict(featurize_smiles(smiles, n_workers=workers, **seed_args))
        assert actual.keys() == reference.keys()
        for smi, expected in reference.items():
            for field in expected:
                np.testing.assert_array_equal(actual[smi][field], expected[field])


@pytest.mark.parametrize("failure", ["inchi", "worker"])
def test_smiles_fallback_preserves_requested_seed(monkeypatch, failure):
    smi = "CCCCO"
    expected = graphs._build_from_smiles_fallback(smi, conformer_seed=123)
    assert expected is not None

    def fail(*args, **kwargs):
        raise ValueError("Force the SMILES fallback")

    if failure == "inchi":
        monkeypatch.setattr(graphs, "build_single_graph", fail)
    else:
        monkeypatch.setattr(graphs, "build_mol", fail)
    actual = _featurize_one(smi, conformer_seed=123)
    assert actual is not None
    np.testing.assert_array_equal(actual[1]["pos_rdkit"], expected["pos_rdkit"])


def test_embedding_seed_controls_conformers():
    # A small real encoder keeps this test independent of downloaded checkpoints.
    torch.manual_seed(42)
    encoder = _build_encoder({"encoder": {
        "rbf_dim": 8, "walk_len": 4, "emb_dim": 8,
        "hidden_dim": 32, "num_heads": 4, "num_layers": 1,
    }}).eval()
    smiles = ["CCCCO", "CCCOC", "CCNCC", "CC(=O)O"]
    a = embed_smiles(smiles, encoder, device="cpu", n_workers=1, conformer_seed=123)
    b = embed_smiles(smiles, encoder, device="cpu", n_workers=4, conformer_seed=123)
    c = embed_smiles(smiles, encoder, device="cpu", n_workers=1, conformer_seed=456)
    assert a.keys() == b.keys() == c.keys() == set(smiles)
    for smi in smiles:
        np.testing.assert_array_equal(a[smi], b[smi])
    assert any(not np.array_equal(a[smi], c[smi]) for smi in smiles)


def test_graph_preprocessing_accepts_conformer_seed(tmp_path, monkeypatch):
    smi = "CCCCO"
    monkeypatch.setattr(graphs, "get_all_smiles", lambda benchmark: [smi])
    monkeypatch.setattr(sys, "argv", [
        "build_eval_graphs", "--output-path", str(tmp_path), "--conformer-seed", "123",
    ])
    graphs.main()
    actual = pq.read_table(tmp_path / "polaris_no_sdf.parquet").to_pylist()[0]
    expected = graphs.build_mol(smi, conformer_seed=123)
    np.testing.assert_array_equal(actual["pos_in"], expected["pos_rdkit"])
