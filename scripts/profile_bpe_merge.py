import argparse
import cProfile
import io
import multiprocessing
import pstats
import time
from pathlib import Path

from cs336_basics.bpe import count_characters_parallel, count_characters_sequential, learn_byte_pairs
from cs336_basics.pretokenization_example import find_chunk_boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the BPE merge-learning phase.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--max-vocab", type=int, default=1000)
    parser.add_argument("--desired-num-chunks", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--sequential-counting", action="store_true")
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=Path("outputs/profile_bpe_merge.prof"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    initial_vocab = 256 + 1

    args.profile_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"input_path={args.input_path}")
    print(f"desired_num_chunks={args.desired_num_chunks}")
    print(f"max_vocab={args.max_vocab}")
    print(f"num_workers={args.num_workers}")

    with args.input_path.open("rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=args.desired_num_chunks,
            split_special_token=b"<|endoftext|>",
        )

    chunks = tuple((start, end, args.input_path) for start, end in zip(boundaries[:-1], boundaries[1:]))
    print(f"actual_chunks={len(chunks)}")

    count_start = time.perf_counter()
    if args.sequential_counting:
        total_counts = count_characters_sequential(chunks)
    else:
        total_counts = count_characters_parallel(chunks, args.num_workers)
    count_elapsed = time.perf_counter() - count_start

    print(f"counting_elapsed_seconds={count_elapsed:.3f}")
    print(f"unique_pretoken_keys={len(total_counts)}")

    profiler = cProfile.Profile()
    merge_start = time.perf_counter()
    profiler.enable()
    merges = learn_byte_pairs(total_counts, initial_vocab, max_vocab=args.max_vocab)
    profiler.disable()
    merge_elapsed = time.perf_counter() - merge_start
    profiler.dump_stats(args.profile_path)

    print(f"merge_elapsed_seconds={merge_elapsed:.3f}")
    print(f"num_merges={len(merges)}")
    print(f"profile_path={args.profile_path}")

    stats_stream = io.StringIO()
    pstats.Stats(profiler, stream=stats_stream).strip_dirs().sort_stats("cumtime").print_stats(30)
    print(stats_stream.getvalue())


if __name__ == "__main__":
    main()
