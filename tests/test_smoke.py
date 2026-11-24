import importlib

import pytest

SUBPACKAGES = ["data", "eval", "generate", "ingest", "retrieve"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name):
    importlib.import_module(f"crag.{name}")
