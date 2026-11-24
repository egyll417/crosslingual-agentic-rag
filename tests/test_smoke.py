import importlib

import pytest

SUBPACKAGES = ["asr", "data"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_importable(name):
    importlib.import_module(f"crag.{name}")
