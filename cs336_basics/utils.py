BytePair = tuple[bytes, bytes]


def merge_word_bytes(
    word: tuple[bytes, ...], merge_candidate: BytePair
) -> tuple[bytes, ...]:
    merged_pseudo_word = []
    merge_candidate_bytes = b"".join(merge_candidate)
    i, word_len = 0, len(word)
    while i < word_len:
        if i + 1 < word_len and (word[i], word[i + 1]) == merge_candidate:
            merged_pseudo_word.append(merge_candidate_bytes)
            # Skip the next element as it's already merged
            i += 2
        else:
            merged_pseudo_word.append(word[i])
            i += 1
    return tuple(merged_pseudo_word)
