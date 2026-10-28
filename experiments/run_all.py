"""Run every experiment notebook in order and save the outputs inside each notebook."""
import glob
import os

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))

for path in sorted(glob.glob(os.path.join(HERE, "[0-9][0-9]_*.ipynb"))):
    print("running", os.path.basename(path))
    notebook = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": HERE}})
    nbformat.write(notebook, path)
print("done - open any notebook to see its results")
