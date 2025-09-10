import json

# Path to your full tokenized dataset
input_file = "tokenized_dataset.jsonl"

# Output file for the smaller dataset
output_file = "100M_dataset.jsonl"

# Target number of tokens
target_tokens = 100_000_000

current_tokens = 0

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        tokens = data.get("input_ids", [])
        if current_tokens + len(tokens) > target_tokens:
            # If adding this line exceeds target, truncate it
            remaining = target_tokens - current_tokens
            if remaining > 0:
                data["input_ids"] = tokens[:remaining]
                outfile.write(json.dumps(data) + "\n")
            break
        else:
            outfile.write(line)
            current_tokens += len(tokens)

print(f"Finished! Total tokens written: {min(current_tokens, target_tokens)}")
