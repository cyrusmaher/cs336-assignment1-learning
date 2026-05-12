# CS336 Assignment 1 Experiment Log

Use one entry per run. Record failed runs too; they are usually the most useful for retention.

## Run Template

### Run ID

- Date:
- Dataset:
- Tokenizer:
- Model config:
- Optimizer config:
- Batch size:
- Context length:
- Total steps or tokens:
- Device:
- Command:
- Git commit:

### Hypothesis

> TODO

### Results

- Final train loss:
- Final validation loss:
- Best validation loss:
- Runtime:
- Peak memory:
- Generated sample path:
- Learning curve path:

### Observations

> TODO

### Next Change

> TODO

## Runs

Add new runs below this line.

### 2026-05-12-bpe-tinystories-5m-pilot

- Date: 2026-05-12
- Dataset: `tests/fixtures/tinystories_sample_5M.txt`
- Tokenizer: byte-level BPE, vocab size 10,000, special token `<|endoftext|>`
- Model config: N/A
- Optimizer config: N/A
- Batch size: N/A
- Context length: N/A
- Total steps or tokens: 9,743 learned merges
- Device: CPU
- Command: `/usr/bin/time -l uv run python -c "... train_bpe(... vocab_size=10000 ...)"`
- Git commit: `e811c62`

### Hypothesis

> Pilot the full TinyStories tokenizer settings on the 5M fixture before launching the 2.1G run.

### Results

- Final train loss: N/A
- Final validation loss: N/A
- Best validation loss: N/A
- Runtime: 35.10s wall-clock
- Peak memory: 47,906,816 bytes maximum resident set size
- Generated sample path: N/A
- Learning curve path: N/A
- Longest token: `b' congratulations'` (16 bytes)

### Observations

> Line profiling the same pilot took 200.97s due to profiler overhead. In the profiled run, `train_bpe` spent 99.4% of time in `learn_byte_pairs`; inside that function, `get_affected_keys` took 53.2%, full `count_pairs` recomputation took 38.2%, and `dict_idxmax` took 8.3%.

### Next Change

> Do not launch the full 2.1G TinyStories tokenizer run until the merge loop is improved or a smaller extrapolation suggests the runtime is acceptable.
