# TinyMamba

TinyMamba is a minimal, educational implementation of a hybrid language model architecture combining state-space modeling (SSM) and transformer attention. It is designed for clarity, extensibility, and experimentation with memory-based reasoning and context-dependent computation.

## Overview

TinyMamba implements a compact neural language model that mixes ideas from the **Mamba** selective state space model and standard **Transformer** blocks.
It includes:

* A **state-space token mixer** for sequential processing and implicit positional encoding
* An optional **Transformer attention layer** for global context
* **Persistent and decaying internal states** that carry memory across generations
* **Context-dependent gating**, allowing the model to modulate how new information updates its internal state

The model can save and reload its internal state between sessions, effectively allowing it to preserve long-term context or gradually forget old information through a decay mechanism.

## Features

* Simple modular code structure using PyTorch
* Hybrid Mamba–Transformer block design
* Configurable decay rate for temporal memory
* Optional goal conditioning via contextual gating
* Training-ready setup with AdamW optimizer
* Persistent memory saving to `./state/` directory

## Usage

```bash
uv run tiny_mamba.py   # You should see 'Tiny Mamba!' and the REPL will start
```

## Intended Purpose

TinyMamba is not meant as a production model.
It serves as a **learning scaffold** for exploring:

* State-space architectures for sequence modeling
* Hybridization of recurrent and attention-based computation
* Persistent and goal-conditioned memory mechanisms in language models

## Research Motivation

TinyMamba is part of an ongoing exploration into **Goal-Conditioned State Space Reasoners (GSSR)**, a theoretical architecture that maintains evolving internal states influenced by both recent inputs and explicit goals.
The aim is to understand how stateful architectures can extend the reasoning horizon of language models without relying solely on external memory or windowed context.

By combining **differentiable memory**, **context-dependent gating**, and **hybrid attention mechanisms**, TinyMamba provides a conceptual foundation for studying how long-term, self-updating state representations can be used to guide generative reasoning.
