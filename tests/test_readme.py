"""Run the Python snippets in README.md so the documented usage stays correct."""

import importlib.util
import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
BLOCKS = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)

pytestmark = pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")


def test_readme_snippets_run():
    namespace: dict = {}
    for block in BLOCKS:
        if "medt.gmre" in block and importlib.util.find_spec("qics") is None:
            continue
        exec(compile(block, str(README), "exec"), namespace)
