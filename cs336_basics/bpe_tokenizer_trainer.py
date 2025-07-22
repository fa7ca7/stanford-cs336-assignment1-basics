import dataclasses
import itertools
import os
import pathlib
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import regex as re

from .pretokenization_example import find_chunk_boundaries
from .utils import BytePair, merge_word_bytes

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOF_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_SPECIAL_TOKENS = [EOF_SPECIAL_TOKEN]


@dataclasses.dataclass
class Word:
    data: tuple[bytes, ...]
    count: int


def pretokenize(
    input_path: str | pathlib.Path,
    start: int,
    end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    path = pathlib.Path(input_path)
    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    pattern = "|".join(map(re.escape, special_tokens))
    words = (
        match.group()
        for text_part in re.split(pattern, chunk)
        for match in re.finditer(PAT, text_part)
    )
    vocab = Counter(words)
    return {
        tuple(bytes([x]) for x in k.encode("utf-8")): v
        for k, v in vocab.items()
    }


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


def compute_byte_pairs(
    pretokenized_corpus: list[Word],
) -> tuple[Counter[BytePair], dict[BytePair, set[int]]]:
    byte_pairs = Counter()
    word_pos = defaultdict(set)
    for i, word in enumerate(pretokenized_corpus):
        data = word.data
        count = word.count
        for j in range(len(data) - 1):
            p = (data[j], data[j + 1])
            byte_pairs[p] += count
            word_pos[p].add(i)
    return byte_pairs, word_pos


def merge_byte_pair(
    pretokenized_corpus: list[Word],
    byte_pairs: Counter[BytePair],
    word_pos: dict[BytePair, set[int]],
    merge_candidate: BytePair,
) -> None:
    affected_word_ids = word_pos.pop(merge_candidate, [])
    for word_id in affected_word_ids:
        word = pretokenized_corpus[word_id]
        data = word.data
        count = word.count
        new_data = merge_word_bytes(data, merge_candidate)
        if data != new_data:
            word.data = new_data
            for j in range(len(data) - 1):
                p = (data[j], data[j + 1])
                byte_pairs[p] -= count
                word_pos[p].discard(word_id)
            for j in range(len(new_data) - 1):
                p = (new_data[j], new_data[j + 1])
                byte_pairs[p] += count
                word_pos[p].add(word_id)


def train_bpe_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[BytePair]]:
    if EOF_SPECIAL_TOKEN not in special_tokens:
        special_tokens.append(EOF_SPECIAL_TOKEN)
    seeds = [st.encode("utf-8") for st in special_tokens]
    seeds += [bytes([x]) for x in range(256)]
    len_seeds = len(seeds)
    pretokenized_map = pretokenize_parallel(input_path, special_tokens)
    pretokenized_corpus = [Word(d, c) for d, c in pretokenized_map.items()]
    merges = []
    byte_pairs, word_pos = compute_byte_pairs(pretokenized_corpus)
    while len_seeds + len(merges) < vocab_size:
        max_occurance, merge_candidate = max(
            (c[1], c[0]) for c in byte_pairs.most_common()
        )
        if max_occurance <= 0:
            break
        merges.append(merge_candidate)
        merge_byte_pair(
            pretokenized_corpus, byte_pairs, word_pos, merge_candidate
        )
    seeds += [b"".join(mc) for mc in merges]
    vocab = dict(enumerate(seeds))
    return vocab, merges
