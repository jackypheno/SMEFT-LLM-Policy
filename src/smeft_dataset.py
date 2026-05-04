import torch

MAX_LEN = 64

class SMEFTDataset:

    def __init__(self, file, vocab):

        self.stoi = {t:i for i,t in enumerate(vocab)}
        self.data = []

        with open(file) as f:
            for line in f:

                line = line.strip()

                # 🚨 skip empty lines
                if len(line) == 0:
                    continue

                toks = line.split()

                # 🚨 skip unknown tokens
                if not all(t in self.stoi for t in toks):
                    continue

                ids = [self.stoi[t] for t in toks]

                # 🚨 ensure non-empty after processing
                if len(ids) == 0:
                    continue

                # truncate
                ids = ids[:MAX_LEN]

                # pad
                while len(ids) < MAX_LEN:
                    ids.append(self.stoi["PAD"])

                self.data.append(torch.tensor(ids, dtype=torch.long))

        # 🚨 FINAL SAFETY CHECK
        assert len(self.data) > 0, "Dataset is empty! Check vocab or train.txt"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].long()
