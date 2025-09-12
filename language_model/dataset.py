import json
from torch.utils.data import Dataset

class SinhalaTokenDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                self.samples.append(example['input_ids'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'input_ids': item,
            'labels': item.copy()  # causal language modeling: predict next token
        }
