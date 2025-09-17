#!/usr/bin/env python3
"""
Transformer Model Parameter Calculator
Calculates total parameters and architecture-only parameters for a given configuration.
"""

def calculate_transformer_parameters(config):
    """
    Calculate parameters for a transformer model given configuration.
    
    Args:
        config (dict): Model configuration containing:
            - vocab_size: Vocabulary size
            - n_positions: Maximum sequence length
            - n_embd: Embedding dimension
            - n_layer: Number of transformer layers
            - n_head: Number of attention heads
    
    Returns:
        dict: Parameter counts breakdown
    """
    
    vocab_size = config["vocab_size"]
    n_positions = config["n_positions"]
    n_embd = config["n_embd"]
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    
    # Embedding parameters
    token_embeddings = vocab_size * n_embd
    position_embeddings = n_positions * n_embd
    embedding_params = token_embeddings + position_embeddings
    
    # Per-layer parameters
    # Multi-head attention
    # Q, K, V projections: 3 * (n_embd * n_embd)
    # Output projection: n_embd * n_embd
    attention_params_per_layer = 4 * (n_embd * n_embd)
    
    # Layer normalization (before attention)
    # Scale and bias parameters
    ln1_params_per_layer = 2 * n_embd
    
    # Feed-forward network
    # Typically 4x expansion: n_embd -> 4*n_embd -> n_embd
    ff_hidden_size = 4 * n_embd
    ff_params_per_layer = (n_embd * ff_hidden_size) + ff_hidden_size + (ff_hidden_size * n_embd) + n_embd
    
    # Layer normalization (before feed-forward)
    ln2_params_per_layer = 2 * n_embd
    
    # Total per layer
    params_per_layer = attention_params_per_layer + ln1_params_per_layer + ff_params_per_layer + ln2_params_per_layer
    
    # All layers
    all_layers_params = n_layer * params_per_layer
    
    # Final layer normalization
    final_ln_params = 2 * n_embd
    
    # Language model head (if separate from token embeddings)
    # Often tied to token embeddings, so we'll count it separately
    lm_head_params = vocab_size * n_embd
    
    # Total parameters
    total_params = embedding_params + all_layers_params + final_ln_params + lm_head_params
    
    # Architecture-only parameters (excluding embeddings and LM head)
    architecture_params = all_layers_params + final_ln_params
    
    return {
        "embedding_breakdown": {
            "token_embeddings": token_embeddings,
            "position_embeddings": position_embeddings,
            "total_embeddings": embedding_params
        },
        "per_layer_breakdown": {
            "attention_params": attention_params_per_layer,
            "layer_norm_1": ln1_params_per_layer,
            "feedforward_params": ff_params_per_layer,
            "layer_norm_2": ln2_params_per_layer,
            "total_per_layer": params_per_layer
        },
        "total_breakdown": {
            "embeddings": embedding_params,
            "all_layers": all_layers_params,
            "final_layer_norm": final_ln_params,
            "lm_head": lm_head_params,
            "total_parameters": total_params,
            "architecture_only": architecture_params
        },
        "summary": {
            "total_parameters": f"{total_params:,}",
            "architecture_parameters": f"{architecture_params:,}",
            "total_parameters_millions": f"{total_params / 1_000_000:.2f}M",
            "architecture_parameters_millions": f"{architecture_params / 1_000_000:.2f}M"
        }
    }

# Given configuration
config = {
  "vocab_size": 4000,
  "n_positions": 512,
  "n_embd": 384,
  "n_layer": 4,
  "n_head": 6,
  "activation_function": "gelu_new",
  "resid_pdrop": 0.1,
  "embd_pdrop": 0.1,
  "attn_pdrop": 0.1,
  "layer_norm_epsilon": 1e-5,
  "initializer_range": 0.02,
  "bos_token_id": 1,
  "eos_token_id": 2
}

# Calculate parameters
results = calculate_transformer_parameters(config)

# Print results
print("=" * 60)
print("TRANSFORMER MODEL PARAMETER ANALYSIS")
print("=" * 60)
print(f"Configuration:")
print(f"  - Vocabulary Size: {config['vocab_size']:,}")
print(f"  - Max Positions: {config['n_positions']:,}")
print(f"  - Embedding Dimension: {config['n_embd']:,}")
print(f"  - Number of Layers: {config['n_layer']}")
print(f"  - Attention Heads: {config['n_head']}")
print()

print("EMBEDDING PARAMETERS:")
print(f"  - Token Embeddings: {results['embedding_breakdown']['token_embeddings']:,}")
print(f"  - Position Embeddings: {results['embedding_breakdown']['position_embeddings']:,}")
print(f"  - Total Embeddings: {results['embedding_breakdown']['total_embeddings']:,}")
print()

print("PER-LAYER PARAMETERS:")
print(f"  - Attention (Q,K,V,O): {results['per_layer_breakdown']['attention_params']:,}")
print(f"  - Layer Norm 1: {results['per_layer_breakdown']['layer_norm_1']:,}")
print(f"  - Feed-Forward: {results['per_layer_breakdown']['feedforward_params']:,}")
print(f"  - Layer Norm 2: {results['per_layer_breakdown']['layer_norm_2']:,}")
print(f"  - Total per Layer: {results['per_layer_breakdown']['total_per_layer']:,}")
print()

print("TOTAL PARAMETER BREAKDOWN:")
print(f"  - Embeddings: {results['total_breakdown']['embeddings']:,}")
print(f"  - All Layers ({config['n_layer']}x): {results['total_breakdown']['all_layers']:,}")
print(f"  - Final Layer Norm: {results['total_breakdown']['final_layer_norm']:,}")
print(f"  - LM Head: {results['total_breakdown']['lm_head']:,}")
print()

print("=" * 60)
print("FINAL RESULTS:")
print("=" * 60)
print(f"Total Parameters: {results['summary']['total_parameters']} ({results['summary']['total_parameters_millions']})")
print(f"Architecture-Only: {results['summary']['architecture_parameters']} ({results['summary']['architecture_parameters_millions']})")
print()

print("Note: Architecture-only parameters exclude token embeddings, position embeddings, and LM head.")
print("These are the parameters that scale with model depth/width rather than vocabulary size.")