# Interval Bound Propagation (NeurIPS 2019)

## Requirements

You might need [poetry 0.12.11](https://github.com/sdispater/poetry) and [jupyterlab 0.35.4](https://github.com/jupyterlab/jupyterlab) installed on [python 3.7.2](https://www.python.org/).

Clone this repository and create a virtual environment:
```sh
git clone https://github.com/ModarTensai/ibp-neurips19 && cd ibp-neurips19
poetry install --extras jupyter
poetry run python -m ipykernel install --user --name ibp-neurips19
```

## Usage

The two most important directories are `ibp-neurips19` (the package folder) and `playground`. Your package folder is your organized code utilities which you use in your experiments. The playground folder is where you put your Jupyter notebooks and experiments code.

Activate the environment and run it (consider using [tmux 2.1](https://github.com/tmux/tmux)):
```sh
source ./.venv/bin/activate
cd playground && jupyter lab
```

Use [vs code 1.31.1](https://code.visualstudio.com/) to edit your package files. Add any new extension to `.vscode/extensions.json` to make them available to everyone.

Don't install any dependency manually. Instead, use poetry and put them in `pyproject.toml`.
