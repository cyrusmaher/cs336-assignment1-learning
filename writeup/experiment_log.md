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

### 2026-05-12-bpe-tinystories-full

- Date: 2026-05-12
- Problem: `train_bpe_tinystories`
- Dataset: `/Users/cyrusmaher/Documents/CS336/TinyStoriesV2-GPT4-train.txt`
- Dataset size: 2,227,753,162 bytes
- Dataset SHA-256: `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443`
- Tokenizer: byte-level BPE, vocab size 10,000, special token `<|endoftext|>`
- Model config: N/A
- Optimizer config: N/A
- Batch size: N/A
- Context length: N/A
- Total steps or tokens: 9,743 learned merges
- Device: CPU
- Python: 3.12.5 via `uv run`
- Command: `scripts/problem_2_5_train_bpe_tinystories.sh`
- Git commit: `e7ed69d`
- Dirty files before run: none

### Hypothesis

> Run the full TinyStories BPE tokenizer training after validating the script on fixtures and a 50MB slice.

### Results

- Final train loss: N/A
- Final validation loss: N/A
- Best validation loss: N/A
- Runtime: 568.83s wall-clock (`elapsed_seconds=568.313`)
- Peak memory: 2,040,037,376 bytes maximum resident set size
- Generated sample path: N/A
- Learning curve path: N/A
- Vocab path: `outputs/problem_2_5_train_bpe_tinystories/vocab.json`
- Merges path: `outputs/problem_2_5_train_bpe_tinystories/merges.txt`
- Summary path: `outputs/problem_2_5_train_bpe_tinystories/summary.json`
- Actual vocab size: 10,000
- Number of merges: 9,743
- Longest token: `b' responsibility'` (15 bytes), token ID 9,379

### Observations

> The full run completed much faster than the rough 50MB linear extrapolation. Peak memory was about 2.04GB RSS, and the longest token looks like a common TinyStories word with a leading space.

### Next Change

> Use these artifacts for the written `train_bpe_tinystories` response, then continue to profiling and/or the OpenWebText tokenizer experiment.
