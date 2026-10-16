"""Execute the tour notebook end to end."""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
def test_tour_notebook_runs():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")
    nb = nbformat.read(EXAMPLES / "medt_tour.ipynb", as_version=4)
    resources = {"metadata": {"path": str(EXAMPLES)}}
    nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3", resources=resources).execute()
