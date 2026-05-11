# CS336 Assignment 1 Learning Repo

This repository is scaffolded for working through Stanford CS336 Assignment 1 without generated solutions.

The goal is to preserve learning and retention:

- Keep written answers in `writeup/assignment1_responses.md`.
- Keep experiment records in `writeup/experiment_log.md`.
- Put your own implementations under `cs336_basics/`.
- Use `tests/adapters.py` only as glue from the tests to your code.
- Keep datasets, checkpoints, generated token arrays, and run logs out of git.

Public-repo note: the starter code is public, but completed assignment solutions may be governed by course policy. If this repo is published publicly, keep it to scaffolding and notes until you are sure publishing completed work is allowed.

## Suggested Work Order

1. Read the Unicode and tokenizer sections, then draft the written responses before coding.
2. Implement and test BPE training on the small fixtures before touching full datasets.
3. Implement tokenizer encode/decode and memory-efficient iterable encoding.
4. Build model components bottom-up, running the narrow pytest target for each one.
5. Add optimizer, scheduler, data loading, checkpointing, and the training loop.
6. Run TinyStories experiments and record every run in the experiment log.
7. Only after the small loop is stable, move to OpenWebText and leaderboard work.

## Local Commands

```sh
uv run pytest -k test_train_bpe
uv run pytest -k test_tokenizer
uv run pytest -k test_linear
uv run pytest -k test_transformer_lm
uv run pytest -k test_adamw
uv run pytest
```

