#text

cli:
    @uv run python src/fontweaver/cli.py

default:
    just --list

quality:
    @uv run ruff check --fix
    @uv run ruff format

train:
    @uv run python src/fontweaver/train.py 

process:
    @uv run python src/fontweaver/process_dataset.py 

sample:
    @uv run python src/fontweaver/sampler.py 

tb:
    @uv run tensorboard --logdir=logs
