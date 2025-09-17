### Model Configuration
{\
  "vocab_size": 32000,\
  "n_positions": 1024,\
  "n_embd": 768,\
  "n_layer": 8,\
  "n_head": 12,\
  "activation_function": "gelu_new",\
  "resid_pdrop": 0.1,\
  "embd_pdrop": 0.1,\
  "attn_pdrop": 0.1,\
  "layer_norm_epsilon": 1e-5,\
  "initializer_range": 0.02,\
  "bos_token_id": 1,\
  "eos_token_id": 2\
}


### 1. `vocab_size: 4000`

**What it is:** Number of unique tokens in the tokenizer.\
**Effect:** Determines the size of the token embedding matrix (`vocab_size` × `n_embd`). \
Larger vocab → bigger embedding matrix, more memory, potentially better handling of rare words.

**Examples:**

- Small model / synthetic dataset: 4,000
- Standard GPT-2: 50,257
- Multilingual models: 250,000+

> Tip: Embedding parameters scale directly with vocab size. For small datasets, small vocab is better.

### 2. `n_positions: 128`

**What it is:** Maximum sequence length (number of tokens the model can attend to at once).\
**Effect:** Determines position embedding matrix size (`n_positions` × `n_embd`). If your sequences exceed this length, you need truncation or longer sequences.

**Examples:**

- GPT-2 small: 1024
- Longformer / LLaMA: 4096+ for long context
- Tiny toy model: 128

### 3. `n_embd: 384`

**What it is:** Dimension of token embeddings and hidden layers.\
**Effect:** Larger n_embd → bigger model capacity. All attention and MLP layers internally scale with n_embd.

**Examples:**

- GPT-2 small: 768
- GPT-2 medium: 1024
- LLaMA 7B: 4096

### 4. `n_layer: 2`

**What it is:** Number of Transformer blocks. Each block has attention + MLP.\
**Effect:** More layers → deeper model → more expressive power, but more parameters and compute.

**Examples:**

- Tiny toy model: 2-4 layers
- GPT-2 small: 12
- LLaMA 7B: 32

### 5. `n_head: 6`

**What it is:** Number of attention heads per layer. Multi-head attention splits the embedding dimension into n_embd / n_head per head.\
**Effect:** More heads → finer attention patterns. But too many small heads → each head sees too little.

**Examples:**

- Small model: 6-8 heads
- GPT-2 small: 12
- GPT-2 large: 20

### 6. `activation_function: "gelu_new"`

**What it is:** Non-linear function applied in MLP layers.\
**Effect:** Determines how the network transforms intermediate features. `gelu_new` is **GELU** variant used in GPT-3.

**Examples:**

- relu – simple, widely used, sometimes slightly worse for transformers
- gelu – smoother, better for deep transformers
- gelu_new – slightly faster, improved stability

### 7. `resid_pdrop: 0.1`

**What it is:** Dropout probability for residual connections.\
**Effect:** Helps regularize the network to prevent overfitting. Applied after adding residual.

**Examples:**

- Small datasets → 0.1-0.2
- Large datasets → 0.0-0.1 (less regularization needed)

### 8. `embd_pdrop: 0.1`

**What it is:** Dropout on token + position embeddings.\
**Effect:** Regularizes the embeddings. Too high → harms learning, too low → overfitting.

### 9. `attn_pdrop: 0.1`

**What it is:** Dropout applied on attention probabilities.\
**Effect:** Prevents the model from relying too heavily on a few tokens.

### 10. `layer_norm_epsilon: 1e-5`

**What it is:** Small number to prevent division by zero in layer norm.\
**Effect:** Helps numerical stability. Usually very small (~1e-5 to 1e-6).

### 11. `initializer_range: 0.02`

**What it is:** Standard deviation for initializing weights (normal distribution).\
**Effect:** Controls scale of initial weights → affects training stability.

**Examples:**

- 0.02 is typical for GPT
- Smaller for deeper models (e.g., 0.01)

### 12. `bos_token_id` & `eos_token_id`

**What they are:** Special IDs for beginning-of-sequence and end-of-sequence.\
**Effect:** Used during training and generation to signal start and stop.

**Examples:**

- GPT: BOS=50256, EOS=50256 (shared)
- Custom: BOS=1, EOS=2