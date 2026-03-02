from pathlib import Path

import tomllib

import flashattention_triton


def test_package_version_matches_project() -> None:
    project = Path(__file__).parents[1] / "pyproject.toml"
    with project.open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    assert flashattention_triton.__version__ == version


def test_interpreter_selected_without_cuda() -> None:
    import os

    import torch

    if not torch.cuda.is_available():
        assert os.environ["TRITON_INTERPRET"] == "1"
