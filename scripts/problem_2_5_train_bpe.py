import json
import sys
import time
from pathlib import Path

from cs336_basics.bpe import train_bpe


def main() -> None:
    input_path = Path(sys.argv[1])
    vocab_size = int(sys.argv[2])
    desired_num_chunks = int(sys.argv[3])
    output_dir = Path(sys.argv[4])
    special_tokens = ["<|endoftext|>"]

    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        desired_num_chunks=desired_num_chunks,
    )
    elapsed_seconds = time.time() - start

    vocab_path = output_dir / "vocab.json"
    merges_path = output_dir / "merges.txt"
    summary_path = output_dir / "summary.json"

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump({str(token_id): repr(token_bytes) for token_id, token_bytes in vocab.items()}, f, indent=2)

    with merges_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            f.write(f"{left!r}\t{right!r}\n")

    longest_token_id, longest_token = max(vocab.items(), key=lambda item: (len(item[1]), item[0]))
    summary = {
        "input_path": str(input_path),
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "elapsed_seconds": elapsed_seconds,
        "longest_token_id": longest_token_id,
        "longest_token_length_bytes": len(longest_token),
        "longest_token_repr": repr(longest_token),
        "vocab_path": str(vocab_path),
        "merges_path": str(merges_path),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"elapsed_seconds={elapsed_seconds:.3f}")
    print(f"vocab_size_actual={len(vocab)}")
    print(f"num_merges={len(merges)}")
    print(f"longest_token_id={longest_token_id}")
    print(f"longest_token_length_bytes={len(longest_token)}")
    print(f"longest_token_repr={longest_token!r}")
    print(f"vocab_path={vocab_path}")
    print(f"merges_path={merges_path}")
    print(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
