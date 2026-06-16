from .ground import all_seqs
from .config import vocab, MAX_LEN

special_tokens = {"START", "END", "UNK", "PAD"}
op_vocab = [v for v in vocab if v not in special_tokens]
SEARCH_SPACE_SIZE, seqs = all_seqs(MAX_LEN, op_vocab)

