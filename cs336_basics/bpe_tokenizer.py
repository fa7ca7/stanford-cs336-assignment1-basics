import itertools
import os
import pathlib
from collections import Counter
from multiprocessing import Pool, cpu_count

import regex as re

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOF_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_SPECIAL_TOKENS = [EOF_SPECIAL_TOKEN]


def pretokenize(
    input_path: str | pathlib.Path, start: int, end: int, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    path = pathlib.Path(input_path)
    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pattern = "|".join(map(re.escape, special_tokens))
    words = [match.group() for text_part in re.split(pattern, chunk) for match in re.finditer(PAT, text_part)]
    vocab = Counter(words)
    return {tuple(map(lambda x: bytes([x]), k.encode("utf-8"))): v for k, v in vocab.items()}


def pretokenize_parallel(
    input_path: str | pathlib.Path,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    path = pathlib.Path(input_path)
    num_processes = cpu_count() - 1
    eof_bytes = EOF_SPECIAL_TOKEN.encode("utf-8")
    with path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, eof_bytes)
    map_args = list(
        zip(
            itertools.repeat(input_path),
            boundaries[:-1],
            boundaries[1:],
            itertools.repeat(special_tokens),
        )
    )
    if len(map_args) == 1:
        results = [pretokenize(*map_args[0])]
    else:
        pool = Pool(processes=num_processes)
        results = pool.starmap(pretokenize, map_args)
    vocab = sum(map(Counter, results), Counter())
    return dict(vocab)


def compute_pretokenized_cache(pretokenized_map: dict[tuple[bytes], int]) -> Counter[tuple[bytes, bytes]]:
    pretokenized_cache = Counter()
    for pseudo_word, count in pretokenized_map.items():
        for left, right in zip(pseudo_word[:-1], pseudo_word[1:]):
            pretokenized_cache[(left, right)] += count
    return pretokenized_cache


def merge_pseudo_word(
    pseudo_word: tuple[bytes, ...],
    merge_candidate: tuple[bytes, bytes],
) -> tuple[bytes, ...]:
    merged_pseudo_word = []
    word_len = len(pseudo_word)
    i = 0
    while i < word_len:
        if i + 1 < word_len and (pseudo_word[i], pseudo_word[i + 1]) == merge_candidate:
            merged_pseudo_word.append(pseudo_word[i] + pseudo_word[i + 1])
            # Skip the next element as it's already merged
            i += 2
        else:
            merged_pseudo_word.append(pseudo_word[i])
            i += 1
    return tuple(merged_pseudo_word)


def merge_pretokenized_map(
    pretokenized_map: dict[tuple[bytes], int],
    merge_candidate: tuple[bytes, bytes],
) -> None:
    for pseudo_word in list(pretokenized_map.keys()):
        merged_pseudo_word = merge_pseudo_word(pseudo_word, merge_candidate)
        if merged_pseudo_word not in pretokenized_map:
            count = pretokenized_map.pop(pseudo_word)
            pretokenized_map[merged_pseudo_word] = count


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if EOF_SPECIAL_TOKEN not in special_tokens:
        special_tokens.append(EOF_SPECIAL_TOKEN)
    seeds = list(map(lambda x: x.encode("utf-8"), special_tokens)) + list(map(lambda x: bytes([x]), range(256)))
    vocab = dict(enumerate(seeds))
    special_tokes_len = len(special_tokens)
    curr_vocab_size = 256 + special_tokes_len
    merges = []
    pretokenized_map = pretokenize_parallel(input_path, special_tokens)
    while curr_vocab_size < vocab_size:
        pretokenized_cache = compute_pretokenized_cache(pretokenized_map)
        max_occurance = max(pretokenized_cache.values() or [0])
        if max_occurance == 0:
            break
        merge_candidate = max(b for b, c in pretokenized_cache.items() if c == max_occurance)
        merges.append(merge_candidate)
        vocab[curr_vocab_size] = b"".join(merge_candidate)
        curr_vocab_size += 1
        merge_pretokenized_map(pretokenized_map, merge_candidate)
    return vocab, merges
