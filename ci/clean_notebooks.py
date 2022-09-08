import json
import pathlib

import nbformat


def clean_notebook(file: pathlib.Path):
    # Read the notebook
    with open(file) as f:
        nb = nbformat.read(f, as_version=4)

    # Remove the output and metadata from each cell
    # also reset the execution count
    # if the cell has no code, remove it
    for cell in nb.cells:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'metadata' in cell:
            cell['metadata'] = {}
        if 'execution_count' in cell:
            cell['execution_count'] = None
        if cell['cell_type'] == 'code' and not cell['source']:
            nb.cells.remove(cell)

    # Write the notebook
    with open(file, 'w') as f:
        nbformat.write(nb, f)


if __name__ == '__main__':
    for file in pathlib.Path.cwd().glob('**/*.ipynb'):
        print(f'Cleaning {file}')
        clean_notebook(file)