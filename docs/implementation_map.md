# Assignment 1 Implementation Map

This map names the files to fill in by hand. It intentionally avoids implementation details.

## Tokenization

- `cs336_basics/bpe.py`: BPE training.
- `cs336_basics/tokenizer.py`: tokenizer construction, encoding, iterable encoding, and decoding.
- Test adapters: `run_train_bpe`, `get_tokenizer`.
- Focused tests: `tests/test_train_bpe.py`, `tests/test_tokenizer.py`.

## Model Components

- `cs336_basics/layers.py`: linear layer, embedding layer, RMSNorm, SiLU/SwiGLU feed-forward pieces.
- `cs336_basics/attention.py`: RoPE, softmax, scaled dot-product attention, multi-head self-attention.
- `cs336_basics/transformer.py`: Transformer block and Transformer LM.
- Test adapters: `run_linear`, `run_embedding`, `run_rmsnorm`, `run_silu`, `run_swiglu`, `run_rope`, `run_scaled_dot_product_attention`, `run_multihead_self_attention`, `run_multihead_self_attention_with_rope`, `run_transformer_block`, `run_transformer_lm`.
- Focused tests: `tests/test_model.py`.

## Training Utilities

- `cs336_basics/nn_utils.py`: softmax, cross-entropy, gradient clipping.
- `cs336_basics/optim.py`: AdamW and cosine learning-rate schedule.
- `cs336_basics/data.py`: batch sampling from token arrays.
- `cs336_basics/checkpointing.py`: save/load model and optimizer state.
- `cs336_basics/training.py`: configurable training loop.
- `cs336_basics/generation.py`: decoding and sampling utilities.
- Test adapters: `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`, `get_adamw_cls`, `run_get_lr_cosine_schedule`, `run_get_batch`, `run_save_checkpoint`, `run_load_checkpoint`.
- Focused tests: `tests/test_nn_utils.py`, `tests/test_optimizer.py`, `tests/test_data.py`, `tests/test_serialization.py`.

## Experiment Artifacts

- `writeup/experiment_log.md`: run records and observations.
- `writeup/assignment1_responses.md`: written response stubs.
- Keep large data, checkpoints, token arrays, and logging output ignored by git.

## Guardrails

- Do not put substantive assignment logic in `tests/adapters.py`; it should only call your implementation.
- Do not edit test files.
- Use the assignment handout as the source of truth for allowed PyTorch APIs.
- Prefer small tests and toy inputs before full dataset runs.

