import torch
import torch.nn as nn
import torch.nn.functional as F

class SMEFTPolicy(nn.Module):
    def __init__(self, vocab_size, max_len=128, d_model=256, nhead=4, nlayers=4):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True
        )

        self.tr = nn.TransformerEncoder(layer, nlayers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = x.long()

        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)

        h = self.emb(x) + self.pos(positions)
        h = self.tr(h)
        h = self.ln(h)

        return self.head(h)

@torch.no_grad()
def generate(model, grammar, prompt, max_len=30):

    model.eval()

    seq = prompt.long()

    for _ in range(max_len):

        logits = model(seq)
        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        nxt = torch.multinomial(probs, 1)

        seq = torch.cat([seq, nxt], dim=1)

    return seq
