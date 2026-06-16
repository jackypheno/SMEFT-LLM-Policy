from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import Counter

def all_seqs(length: int, vocab: list) -> tuple[int, list[tuple]]:
    """
    Returns all combinations (no repetition, order doesn't matter)
    of operators up to the given length.

    Args:
        length: maximum sequence length (inclusive), starting from 1
        vocab:  list of operator tokens

    Returns:
        (count, sequences)  where sequences is a list of tuples
    """
    sequences = []
    for k in range(1, length + 1):
        for seq in combinations(vocab, k):
            sequences.append(seq)
    return len(sequences), sequences


