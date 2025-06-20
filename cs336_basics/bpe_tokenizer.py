import itertools
import pathlib
from collections import Counter
from multiprocessing import Pool, cpu_count

import regex as re

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOF_SPECIAL_TOKEN = "<|endoftext|>"
DEFAULT_SPECIAL_TOKENS = [EOF_SPECIAL_TOKEN]


def pretokenize(input_path: str | pathlib.Path, start: int, end: int) -> dict[tuple[bytes], int]:
    path = pathlib.Path(input_path)
    with path.open("rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    vocab = Counter(tuple(match.group().encode("utf-8")) for match in re.finditer(PAT, chunk))
    return dict(vocab)


def pretokenize_parallel(
    input_path: str | pathlib.Path,
) -> dict[tuple[bytes], int]:
    path = pathlib.Path(input_path)
    num_processes = cpu_count() - 1
    eof_bytes = EOF_SPECIAL_TOKEN.encode("utf-8")
    with path.open("rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, eof_bytes)
    args = zip(itertools.repeat(input_path), boundaries[:-1], boundaries[1:])
    pool = Pool(processes=num_processes)
    results = pool.starmap(pretokenize, args)
    vocab = sum(map(Counter, results), Counter())
    return dict(vocab)
