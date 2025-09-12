from transformers import GPT2LMHeadModel
import torch
from torch.utils.data import DataLoader
from dataset import SinhalaTokenDataset  # reuse your dataset class
from collator import CustomDataCollator  # reuse your collator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = GPT2LMHeadModel.from_pretrained("final-model").to(device)
model.eval()

# Load test dataset
test_dataset = SinhalaTokenDataset("")  # 500K test set
collator = CustomDataCollator(pad_token_id=0)
loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collator)

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        num_tokens = attention_mask.sum().item()

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

# Compute Perplexity
avg_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(avg_loss))

print(f"Test Perplexity: {perplexity.item():.2f}")
