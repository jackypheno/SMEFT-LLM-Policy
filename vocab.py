# ============================================================
# smeft_rl/vocab.py
# Vocabulary loading and sequence parsing utilities.
# ============================================================

from pathlib import Path
from .config import VOCAB_PATH, DEVICE

vocab = Path(VOCAB_PATH).read_text().split()
stoi  = {t: i for i, t in enumerate(vocab)}
itos  = {i: t for t, i in stoi.items()}

PAD_ID   = stoi["PAD"]
START_ID = stoi["START"]
END_ID   = stoi["END"]
UNK_ID   = stoi["UNK"]

INVALID = {"UNK"}
SPECIAL = {"PAD", "START", "END"}


def seq_to_tokens(seq):
    """Convert a token ID tensor to a list of string tokens up to END."""
    tokens = []
    for i in seq.detach().cpu().tolist():
        t = itos.get(i, "UNK")
        tokens.append(t)
        if t == "END":
            break
    return tokens


def extract_ops(tokens):
    """
    Parse token list into (deduped_ops, raw_ops, terminated).
    Skips INVALID and SPECIAL tokens.
    """
    raw_ops    = []
    terminated = False
    started    = False
    for t in tokens:
        if t in INVALID:
            continue
        if t == "START":
            started = True
            continue
        if t == "END":
            terminated = True
            break
        if started and t not in SPECIAL:
            raw_ops.append(t)
    return list(dict.fromkeys(raw_ops)), raw_ops, terminated
