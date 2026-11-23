"""Minimal inference example: featurize one molecule and embed it.

``load_ckpt`` restores the encoder only, so a forward pass returns Monroe's
graph-level embedding rather than task predictions. That embedding is the thing
you feed to a downstream head — it is exactly what the evaluation pipeline caches
before fitting TabPFN or an MLP on top (see monroe/eval/).

Usage:
    python monroe/run_inference.py --ckpt-path /path/to/checkpoint
    python monroe/run_inference.py --ckpt-path ... --smiles "CC(=O)Oc1ccccc1C(=O)O"

A checkpoint directory is one containing config.json plus weights.pt (or
ema_weights.pt, selected with --use-ema). With no --smiles, a molecule is pulled
from a Polaris benchmark, which requires the [eval] extra.
"""
import argparse
import sys

import datamol as dm
import numpy as np
import torch
from torch_geometric.data import Data

from monroe.checkpoints import add_checkpoint_arguments
from monroe.model.ckpt import load_ckpt
from monroe.model.featurizer import DEFAULT_CONFORMER_SEED, build_single_graph


def is_flat(pos: np.ndarray) -> bool:
    """True if every atom sits at z=0, i.e. a 2D layout rather than a conformer."""
    return pos.ndim == 2 and pos.shape[0] > 0 and bool(np.all(pos[:, 2] == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_checkpoint_arguments(parser)
    parser.add_argument("--smiles", default=None,
                        help="Molecule to embed. Defaults to one from a Polaris benchmark.")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_weights.pt instead of weights.pt.")
    parser.add_argument("--conformer-seed", type=int, default=DEFAULT_CONFORMER_SEED,
                        help="RDKit conformer seed (default: %(default)s).")
    args = parser.parse_args()

    encoder = load_ckpt(args.ckpt_path, use_ema=args.use_ema, model=args.model)
    encoder.eval()

    smi = args.smiles
    if smi is None:
        import polaris as po
        benchmark = po.load_benchmark("polaris/adme-fang-hppb-1")
        _, test_split = benchmark.get_train_test_split()
        smi = test_split[-1]

    inchi = dm.to_inchi(smi)
    if inchi is None:
        sys.exit(f"RDKit could not parse SMILES: {smi!r}")

    # Coordinates come from RDKit (ETKDG + MMFF94s, falling back to UFF). Monroe
    # is trained to work from these rather than needing the PM6-optimised geometry.
    try:
        graph_dict = build_single_graph(inchi=inchi, symmetrize=True, conformer_seed=args.conformer_seed)
    except ValueError as exc:
        sys.exit(f"Could not featurize {smi!r}: {exc}")

    pos = np.asarray(graph_dict["pos_rdkit"], dtype=np.float32)
    if is_flat(pos):
        # build_single_graph falls back to Compute2DCoords when conformer
        # generation fails, which yields a planar molecule.
        print(
            f"WARNING: 3D conformer generation failed for {smi!r}; the featurizer fell\n"
            f"         back to a flat 2D layout (all z=0). Monroe is structure-aware, so\n"
            f"         this embedding is unreliable. scripts/preprocessing/fix_2d_molecules.py\n"
            f"         holds the retry-with-relaxed-parameters logic used to repair such\n"
            f"         molecules in bulk.",
            file=sys.stderr,
        )

    data = Data(
        x=torch.tensor(graph_dict["node_float"], dtype=torch.float32),
        node_codes=torch.tensor(graph_dict["node_codes"], dtype=torch.long),
        edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
        edge_codes=torch.tensor(graph_dict["edge_codes"], dtype=torch.long),
        pos_in=torch.tensor(pos, dtype=torch.float32),
    )

    device = next(encoder.parameters()).device
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(device)

    with torch.no_grad():
        graph_emb, _ = encoder(data)

    print(f"Molecule:        {smi}")
    print(f"Atoms:           {data.x.size(0)}")
    print(f"Geometry:        {'2D fallback (unreliable)' if is_flat(pos) else '3D conformer'}")
    print(f"Graph embedding: {tuple(graph_emb.shape)}")
    print(graph_emb)


if __name__ == "__main__":
    main()
