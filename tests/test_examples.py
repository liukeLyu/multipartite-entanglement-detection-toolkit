"""Execute the example notebooks end to end."""

from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["medt_tour.ipynb", "medt_fermions.ipynb"])
def test_notebook_runs(name):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("matplotlib")
    nb = nbformat.read(EXAMPLES / name, as_version=4)
    resources = {"metadata": {"path": str(EXAMPLES)}}
    nbclient.NotebookClient(nb, timeout=1200, kernel_name="python3", resources=resources).execute()
