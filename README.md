# HKUST Ising Caviar

A small Python project for exploring the Ising model and related computations.

This project uses `pyproject.toml` to declare dependencies and relies on `uv` to manage the Python environment and install those dependencies.


## Prerequisites

- Python 3.13 or later
- `uv` installed globally or in a separate bootstrap environment

## Install `uv`

Install `uv` with one of the following methods:

```bash
python -m pip install --upgrade uv
```


## Install dependencies

From the project root, run:

```bash
uv sync
```

This reads the dependency list from `pyproject.toml` and creates or updates the project environment.

## Run the project

Use `uv run` to execute the project within the managed environment:

```bash
uv run v2.1.2.py
```


Then you can run Python commands directly.

## Add or remove dependencies

To add a dependency:

```bash
uv add <package-name>
```

To remove a dependency:

```bash
uv remove <package-name>
```

## Notes

- The dependencies are declared in `pyproject.toml`.
- Use `uv sync` after editing dependencies to keep the environment synced.
- `uv` is used in this project for both dependency installation and runtime execution.
