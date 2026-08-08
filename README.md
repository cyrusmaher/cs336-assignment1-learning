# CS336 Spring 2025 Assignment 1: Basics

## Implementation status

This is my in-progress implementation of Stanford CS336 Assignment 1. The goal
is to build a decoder-only Transformer language model and its data and training
stack from low-level PyTorch tensor operations rather than use high-level model
or training frameworks.

Completed work currently includes:

- byte-level BPE training, including a recorded full 2.2 GB TinyStories
  tokenizer run;
- linear and embedding layers, RMSNorm, RoPE, scaled dot-product attention,
  causal multi-head attention, SwiGLU blocks, Transformer blocks, and an
  assembled Transformer language model; and
- experiment scripts and provenance records for the completed tokenizer runs.

The repository is not yet a completed end-to-end training stack. As of the
2026-08-07 audit, the principal Transformer assembly tests pass, while the
tokenizer adapter, SiLU/SwiGLU adapter coverage, optimizer and training
utilities, and full test suite remain in progress. Public descriptions should
therefore say that I am *implementing* the stack, not that I have completed it.

See [docs/implementation_map.md](docs/implementation_map.md) for the component
map and [writeup/experiment_log.md](writeup/experiment_log.md) for recorded
runs.

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
