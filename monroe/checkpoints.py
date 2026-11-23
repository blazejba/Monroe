"""Selection of bundled models or custom checkpoint directories."""

from pathlib import Path


CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
MODEL_NAMES = ("default", "no_decorr")


def resolve_checkpoint(path: str | Path | None = None, *, model: str | None = None) -> Path:
    """Use the released model by default; explicit paths retain their meaning."""
    if path is not None:
        if model is not None:
            raise ValueError("Choose either a checkpoint path or a model name, not both.")
        return Path(path)
    name = "default" if model is None else model
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}.")
    return CHECKPOINTS_DIR / name


def add_checkpoint_arguments(parser):
    """Expose the same model selection options in inference and evaluation."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model", choices=MODEL_NAMES,
        help="Bundled model to load (default: default).",
    )
    group.add_argument(
        "--ckpt-path",
        help="Custom checkpoint directory or experiment root (uses its latest checkpoint).",
    )
