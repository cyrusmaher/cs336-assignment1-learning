# Student implementation placeholder. Fill this in by working through the BPE training section.
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
from os import path
from collections import defaultdict
from multiprocessing import Pool
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
input_path = path.expanduser('~/Documents/CS336/TinyStoriesV2-GPT4-train.txt')
special_tokens = ["<|endoftext|>"]
special_token_PAT = re.compile(r"(?:" + "|".join(re.escape(token) for token in special_tokens) + r")")

def count_pairs(result_counts):
    pairs_counts = defaultdict(int)
    for kk, value in result_counts.items():
        for i in range(len(kk) - 1):
            pairs_counts[kk[i:i+2]] += value
    return pairs_counts

def string_to_bytes_tuple(string):
    return tuple(bytes([b]) for b in string.encode("utf-8"))

def decode_chunk(start, end, f):
    f.seek(start)
    chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return chunk

def count_characters(chunk):
    result = defaultdict(int)
    for sub_chunk in special_token_PAT.split(chunk):
        for match in re.finditer(PAT, sub_chunk):
            res = match.group()
            result[string_to_bytes_tuple(res)] += 1
    return result

def merge_counts(*counts):
    result = defaultdict(int)
    for counts in counts:
        for key, value in counts.items():
            result[key] += value
    return result

def dict_idxmax(dd):
    return max(dd.items(), key=lambda x: (x[1], x[0]))

def merge_pair_in_key(key, pair):
    found = False
    for this_pair in zip(key[:-1], key[1:]):
        if this_pair == pair:
            found = True
            break
    if not found:
        return key

    merged = None
    new_key = []
    i = 0

    while i < len(key):
        if key[i:i+2] == pair:
            if merged is None:
                merged = pair[0] + pair[1]
            new_key.append(merged)
            i += 2
        else:
            new_key.append(key[i])
            i += 1
    return tuple(new_key)

def _get_affected_keys(key, pair):
    for i in range(len(key) - 1):
        if key[i:i+2] == pair:
            return True
    return False

def get_affected_keys(total_counts, pair):
    affected_keys = {}
    for key, count in total_counts.items():
        if _get_affected_keys(key, pair):
            affected_keys[key] = ""
    return affected_keys

def learn_byte_pairs(total_counts, vocab_size, max_vocab=500):
    to_merge = []

    pair_counts = count_pairs(total_counts)
    while vocab_size < max_vocab:
        if len(pair_counts) == 0:
            break

        max_key, _ = dict_idxmax(pair_counts)
        # We need to maintain a merge list in order. This is what we will take forward to the transformation step
        to_merge.append(max_key)

        # Only mutate affected keys
        affected_keys = get_affected_keys(total_counts, max_key)

        # First update
        for key in affected_keys.keys():
            new_key = merge_pair_in_key(key, max_key)
            total_counts[new_key] += total_counts[key]

        # Then delete the original version of the mutated keys
        for key in affected_keys.keys():
            del total_counts[key]

        pair_counts = count_pairs(total_counts)
        vocab_size += 1

    return to_merge

def build_vocab_dict(special_characters, to_merge, min_vocab_size=256):
    vocab_dict = {}
    for i in range(min_vocab_size):
        vocab_dict[i] = bytes([i])

    for ss in special_characters:
        i += 1
        vocab_dict[i] = ss.encode("utf-8")

    for merge in to_merge:
        i += 1
        vocab_dict[i] = merge[0]+merge[1]
    return vocab_dict


def train_bpe(input_path, vocab_size, special_tokens, desired_num_chunks=20, **kwargs):
    if "num_workers" not in kwargs:
        num_workers = multiprocessing.cpu_count()
    else:
        num_workers = kwargs["num_workers"]

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=desired_num_chunks,
            split_special_token=b"<|endoftext|>",
        )


        chunks = (decode_chunk(start, end, f) for start, end in zip(boundaries[:-1], boundaries[1:]))

        if vocab_size > 1000:
            with Pool(num_workers) as p:
                all_results = p.map(count_characters, chunks)
        else:
            all_results = []
            for chunk in chunks:
                all_results.append(count_characters(chunk))

        total_counts = merge_counts(*all_results)
        to_merge = learn_byte_pairs(total_counts, 256 + len(special_tokens), max_vocab=vocab_size)
        vocab_dict = build_vocab_dict(special_tokens, to_merge)

    return vocab_dict, to_merge

if __name__ == "__main__":

    vocab_dict, to_merge = train_bpe(input_path, vocab_size=500, special_tokens=special_tokens, desired_num_chunks=20)
    print(vocab_dict)
    print(to_merge)