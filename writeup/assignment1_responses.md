# CS336 Assignment 1 Response Stubs

Source: `cs336_assignment1_basics.pdf`, Version 26.0.3, Spring 2026.

Do not fill these with generated answers. Use this as a checklist while you work through the assignment.

## Problem `unicode1`: Understanding Unicode

### (a)

Prompt: What Unicode character does `chr(0)` return?

Deliverable: One-sentence response.

Response:

> `chr(0)` returns a Python string of length 1 whose content is the Unicode null character, represented as `'\x00'`.

### (b)

Prompt: How does this character's string representation differ from its printed representation?

Deliverable: One-sentence response.

Response:

> `repr(chr(0))` displays the null character with escapes, as `'\x00'`, while `print(chr(0))` emits the null character itself, which is not visible.

### (c)

Prompt: What happens when this character occurs in text?

Deliverable: One-sentence response.

Response:

> When it occurs in text, it usually renders invisibly, so the surrounding characters may appear adjacent, but the null character is still present in the string.

## Problem `unicode2`: Unicode Encodings

### (a)

Prompt: Compare UTF-8 bytes with UTF-16 and UTF-32 for tokenizer training.

Deliverable: One-to-two sentence response.

Response:

> UTF-16 and UTF-32 often require more bytes for common text than UTF-8, so they produce longer byte sequences and make tokenizer training and downstream modeling less efficient.

### (b)

Prompt: Explain why decoding each UTF-8 byte independently is incorrect, using an example byte string.

Deliverable: Example input byte string plus one-sentence explanation.

Response:

> For example, `"🙂".encode("utf-8")` produces a multi-byte input such as `b'\xf0\x9f\x99\x82'`, and the function fails because it assumes each byte can be decoded as a complete character even though UTF-8 characters can span multiple bytes.

### (c)

Prompt: Give a two-byte sequence that does not decode to Unicode.

Deliverable: Example plus one-sentence explanation.

Response:

> `b"\xbf\xbf"` is invalid UTF-8 because both bytes are continuation bytes, and a continuation byte cannot start a Unicode character.

## Problem `train_bpe`: BPE Tokenizer Training

Deliverable: Implement byte-level BPE tokenizer training.

Implementation status:

> TODO

Reflection notes:

> TODO: Record design choices, invariants, and performance observations after implementing.

## Problem `train_bpe_tinystories`: BPE Training on TinyStories

### (a)

Prompt: Train a 10,000 vocabulary byte-level BPE tokenizer with the TinyStories special token; report time, memory, and longest token.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (b)

Prompt: Profile tokenizer training and identify the slowest part.

Deliverable: One-to-two sentence response.

Response:

> TODO

## Problem `train_bpe_expts_owt`: BPE Training on OpenWebText

### (a)

Prompt: Train a 32,000 vocabulary byte-level BPE tokenizer on OpenWebText; inspect the longest token.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (b)

Prompt: Compare the TinyStories tokenizer with the OpenWebText tokenizer.

Deliverable: One-to-two sentence response.

Response:

> TODO

## Problem `tokenizer`: Implementing the Tokenizer

Deliverable: Implement a tokenizer class that encodes text to token IDs and decodes IDs to text, including special-token handling and iterable encoding.

Implementation status:

> TODO

Reflection notes:

> TODO

## Problem `tokenizer_experiments`: Experiments with Tokenizers

### (a)

Prompt: Sample 10 documents from TinyStories and OpenWebText, encode them, and compute bytes/token compression ratios.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (b)

Prompt: Tokenize OpenWebText samples with the TinyStories tokenizer and compare behavior.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (c)

Prompt: Estimate tokenizer throughput and extrapolate to the Pile.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (d)

Prompt: Encode training and validation datasets; explain why `uint16` is appropriate.

Deliverable: One-to-two sentence response.

Response:

> TODO

## Problem `linear`: Implementing the Linear Module

Deliverable: Implement a bias-free linear module and adapter.

Implementation status:

> TODO

## Problem `embedding`: Implement the Embedding Module

Deliverable: Implement an embedding module and adapter.

Implementation status:

> TODO

## Problem `rmsnorm`: Root Mean Square Layer Normalization

Deliverable: Implement RMSNorm and adapter.

Implementation status:

> TODO

## Problem `positionwise_feedforward`: Position-Wise Feed-Forward Network

Deliverable: Implement SwiGLU feed-forward network and adapter.

Implementation status:

> TODO

## Problem `rope`: Rotary Positional Embedding

Deliverable: Implement RoPE and adapter.

Implementation status:

> TODO

## Problem `softmax`: Implement Softmax

Deliverable: Implement numerically stable softmax and adapter.

Implementation status:

> TODO

## Problem `scaled_dot_product_attention`: Scaled Dot-Product Attention

Deliverable: Implement scaled dot-product attention with masking and adapter.

Implementation status:

> TODO

## Problem `multihead_self_attention`: Causal Multi-Head Self-Attention

Deliverable: Implement causal multi-head self-attention and adapter.

Implementation status:

> TODO

## Problem `transformer_block`: Transformer Block

Deliverable: Implement a pre-norm Transformer block and adapter.

Implementation status:

> TODO

## Problem `transformer_lm`: Transformer LM

Deliverable: Implement the Transformer language model and adapter.

Implementation status:

> TODO

## Problem `transformer_accounting`: Transformer LM Resource Accounting

### (a)

Prompt: Count trainable parameters and model-load memory for the assignment's GPT-2 XL-shaped configuration.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (b)

Prompt: Identify forward-pass matrix multiplies and total FLOPs for the GPT-2 XL-shaped model at full context length.

Deliverable: List of matrix multiplies with descriptions plus total FLOPs.

Response:

> TODO

### (c)

Prompt: Identify which model parts require the most FLOPs.

Deliverable: One-to-two sentence response.

Response:

> TODO

### (d)

Prompt: Repeat FLOPs breakdown for GPT-2 small, medium, and large; compare proportional changes.

Deliverable: Component breakdowns as proportions plus one-to-two sentence description.

Response:

> TODO

### (e)

Prompt: Increase GPT-2 XL context length to 16,384 and analyze total and relative FLOPs.

Deliverable: One-to-two sentence response.

Response:

> TODO

## Problem `cross_entropy`: Cross-Entropy

Deliverable: Implement numerically stable cross-entropy and adapter.

Implementation status:

> TODO

## Problem `learning_rate_tuning`: Tuning the Learning Rate

Prompt: Run the toy SGD example with larger learning rates and observe loss behavior.

Deliverable: One-to-two sentence response.

Response:

> TODO

## Problem `adamw`: AdamW

Deliverable: Implement AdamW as an optimizer subclass and adapter.

Implementation status:

> TODO

## Problem `adamw_accounting`: Resource Accounting for AdamW

### (a)

Prompt: Derive peak-memory terms for parameters, activations, gradients, optimizer state, and total.

Deliverable: Algebraic expressions.

Response:

> TODO

### (b)

Prompt: Instantiate the memory expression for GPT-2 XL and find maximum batch size under 80GB.

Deliverable: Expression in `a * batch_size + b` form plus maximum batch size.

Response:

> TODO

### (c)

Prompt: Count FLOPs for one AdamW step.

Deliverable: Algebraic expression with brief justification.

Response:

> TODO

### (d)

Prompt: Estimate GPT-2 XL training time on one H100 at 50% MFU for the specified training setup.

Deliverable: Hours plus brief justification.

Response:

> TODO

## Problem `learning_rate_schedule`: Cosine Schedule with Warmup

Deliverable: Implement the cosine learning-rate schedule and adapter.

Implementation status:

> TODO

## Problem `gradient_clipping`: Gradient Clipping

Deliverable: Implement in-place gradient clipping and adapter.

Implementation status:

> TODO

## Problem `data_loading`: Data Loading

Deliverable: Implement batch sampling from token arrays and adapter.

Implementation status:

> TODO

## Problem `checkpointing`: Model Checkpointing

Deliverable: Implement checkpoint save/load functions and adapters.

Implementation status:

> TODO

## Problem `training_together`: Put It Together

Deliverable: Write a configurable training script with large-dataset loading, checkpointing, and logging.

Implementation status:

> TODO

## Problem `decoding`: Decoding

Deliverable: Implement generation from a trained language model with prompt, max-token, temperature, and top-p controls.

Implementation status:

> TODO

## Problem `experiment_log`: Experiment Logging

Deliverable: Logging infrastructure and an experiment log for the experiments below.

Implementation status:

> TODO

## Problem `learning_rate`: Tune the Learning Rate

### (a)

Prompt: Sweep learning rates, report final losses or divergence, and reach the target TinyStories validation loss.

Deliverable: Learning curves, search strategy explanation, and model meeting the target.

Response:

> TODO

### (b)

Prompt: Investigate the relationship between divergent learning rates and the best learning rate.

Deliverable: Learning curves including at least one divergent run plus analysis.

Response:

> TODO

## Problem `batch_size_experiment`: Batch Size Variations

Prompt: Vary batch size from 1 to the memory limit and compare training behavior.

Deliverables: Learning curves and a few sentences discussing findings.

Response:

> TODO

## Problem `generate`: Generate Text

Prompt: Generate text from your trained TinyStories checkpoint and comment on fluency.

Deliverable: At least 256 generated tokens or until `<|endoftext|>`, plus brief commentary and two quality factors.

Response:

> TODO

## Problem `layer_norm_ablation`: Remove RMSNorm and Train

Prompt: Remove RMSNorms, train, and compare stability at different learning rates.

Deliverables: Learning curves and commentary.

Response:

> TODO

## Problem `pre_norm_ablation`: Implement Post-Norm and Train

Prompt: Modify the Transformer to post-norm and compare with pre-norm.

Deliverable: Learning curve comparison.

Response:

> TODO

## Problem `no_pos_emb`: Implement NoPE

Prompt: Remove positional embedding information and compare with RoPE.

Deliverable: Learning curve comparison.

Response:

> TODO

## Problem `swiglu_ablation`: SwiGLU vs. SiLU

Prompt: Compare SwiGLU with a parameter-matched SiLU feed-forward network.

Deliverables: Learning curve comparison and discussion.

Response:

> TODO

## Problem `main_experiment`: Experiment on OpenWebText

Prompt: Train on OpenWebText with the same model architecture and total training iterations as TinyStories.

Deliverables: Learning curve, loss interpretation, generated text, and fluency commentary.

Response:

> TODO

## Problem `leaderboard`: Leaderboard

Prompt: Train under the leaderboard rules with the goal of minimizing validation loss within the time budget.

Deliverable: Final validation loss, learning curve with wall-clock x-axis under the limit, and description of changes.

Response:

> TODO
