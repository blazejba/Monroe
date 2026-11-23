"""Bundled model selection must preserve explicit custom checkpoint paths."""

import argparse

import pytest

from monroe.checkpoints import CHECKPOINTS_DIR, add_checkpoint_arguments, resolve_checkpoint
from monroe.model import ckpt


def test_default_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_checkpoint() == CHECKPOINTS_DIR / "default"
    assert resolve_checkpoint(model="no_decorr") == CHECKPOINTS_DIR / "no_decorr"


def test_custom_experiment_still_resolves_latest_checkpoint(tmp_path):
    for index in (4, 12):
        folder = tmp_path / f"checkpoint-{index}"
        folder.mkdir()
        (folder / "config.json").write_text("{}")
    assert ckpt._resolve_ckpt_dir(resolve_checkpoint(tmp_path)) == tmp_path / "checkpoint-12"


def test_invalid_selection_never_falls_back_to_default():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_checkpoint(model="typo")
    with pytest.raises(ValueError, match="either"):
        resolve_checkpoint("custom", model="no_decorr")


def test_cli_selection():
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser)
    args = parser.parse_args([])
    assert resolve_checkpoint(args.ckpt_path, model=args.model) == CHECKPOINTS_DIR / "default"
    args = parser.parse_args(["--model", "no_decorr"])
    assert resolve_checkpoint(args.ckpt_path, model=args.model) == CHECKPOINTS_DIR / "no_decorr"
    args = parser.parse_args(["--ckpt-path", "custom"])
    assert str(resolve_checkpoint(args.ckpt_path, model=args.model)) == "custom"
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "default", "--ckpt-path", "custom"])


@pytest.mark.parametrize("model", [None, "no_decorr"])
def test_loader_uses_selected_bundle(model, monkeypatch):
    # Stop before reading large artifacts, but exercise the public loader routing.
    class SelectedCheckpoint(Exception):
        pass

    def inspect_path(path, **kwargs):
        assert path == CHECKPOINTS_DIR / (model or "default")
        assert kwargs["use_ema"] is False
        raise SelectedCheckpoint

    monkeypatch.setattr(ckpt, "_load_config_and_weights", inspect_path)
    with pytest.raises(SelectedCheckpoint):
        ckpt.load_ckpt(model=model)
