## Introduction

Attention is the central mechanism powering modern neural architectures such as the **Transformer**. It allows models to dynamically focus on relevant parts of the input sequence, instead of relying on fixed-size context windows (like CNNs) or recursive steps (like RNNs).

At the core of attention are four key components: **Queries (Q), Keys (K), Values (V),** and the resulting **Output (O)**.

## Formal Definitions

Given an input sequence $X ∈ R^{n×d_{model}}$, where $n$ is the sequence length and $d_{model}$ is the embedding dimension.