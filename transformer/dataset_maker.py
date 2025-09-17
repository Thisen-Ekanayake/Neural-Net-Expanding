import random

def create_subset_dataset(input_file, output_file, target_word_count=100_000_000, shuffle=False):
    """
    Create a dataset with a fixed number of words from a larger dataset.

    Args:
        input_file (str): Path to the larger dataset (.txt file).
        output_file (str): Path to save the 100M word dataset (.txt file).
        target_word_count (int): Number of words to extract (default: 100M).
        shuffle (bool): If True, shuffles the collected words before writing.
    """
    total_words = 0
    collected_words = []

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            words = line.strip().split()
            if not words:
                continue

            # If we would exceed the target, trim the list
            if total_words + len(words) > target_word_count:
                needed = target_word_count - total_words
                collected_words.extend(words[:needed])
                total_words += needed
                break
            else:
                collected_words.extend(words)
                total_words += len(words)

            # Print progress every 10M words
            if total_words % 10_000_000 < len(words):
                print(f"Collected {total_words:,} words so far...")

    # Shuffle if needed
    if shuffle:
        print("Shuffling collected words...")
        random.shuffle(collected_words)

    # Save to output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(" ".join(collected_words))

    print(f"Done! Saved {total_words:,} words to {output_file}")


# -------- Usage Example --------
# Replace with your dataset paths
input_path = "All-Text_8696658_147190824.txt"
output_path = "100M_words.txt"

create_subset_dataset(input_path, output_path, target_word_count=100_000_000, shuffle=False)
