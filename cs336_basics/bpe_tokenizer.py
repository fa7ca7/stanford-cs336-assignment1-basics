import dataclasses
import functools
import json
import os
import pathlib
from collections.abc import Iterable, Iterator
from typing import Self

import regex as re

from .utils import PAT, merge_word_bytes


@dataclasses.dataclass
class BPETokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def __post_init__(self) -> None:
        if self.special_tokens is None:
            self.special_tokens = []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: os.PathLike,
        merges_filepath: os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> Self:
        vocab_data = pathlib.Path(vocab_filepath).read_text(encoding="utf-8")
        vocab = json.loads(vocab_data)
        merges_data = pathlib.Path(merges_filepath).read_text(encoding="utf-8")
        merges = json.loads(merges_data)
        return Self(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @functools.cached_property
    def revocab(self) -> dict[bytes, int]:
        return {word: token for token, word in self.vocab.items()}

    @functools.cached_property
    def spetial_tokens_pattern(self) -> str:
        pattern = "|".join(
            map(re.escape, sorted(self.special_tokens, reverse=True))
        )
        return f"({pattern})"

    def encode(self, text: str) -> list[int]:
        tokens = []
        text_gen = (
            [text]
            if not self.special_tokens
            else re.split(self.spetial_tokens_pattern, text)
        )
        for chunk in text_gen:
            if chunk in self.special_tokens:
                # Most likely special tokens
                byte_seq = chunk.encode("utf-8")
                if token := self.revocab.get(byte_seq):
                    tokens.append(token)
            else:
                for word in re.finditer(PAT, chunk):
                    byte_seq = word.group().encode("utf-8")
                    byte_word = tuple(bytes([b]) for b in byte_seq)
                    for merge_candidate in self.merges:
                        byte_word = merge_word_bytes(
                            byte_word, merge_candidate
                        )
                    for b in byte_word:
                        tokens.append(self.revocab[b])
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.vocab[i] for i in ids)
        return byte_string.decode("utf-8", errors="replace")
