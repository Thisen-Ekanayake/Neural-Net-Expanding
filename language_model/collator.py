import torch
from torch.nn.utils.rnn import pad_sequence

class CustomDataCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        padded_inputs = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = (padded_inputs != self.pad_token_id).long()

        labels = padded_inputs.clone()
        return {
            "input_ids": padded_inputs,
            "attention_mask": attention_mask,
            "labels": labels
        }
