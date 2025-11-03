# ⚙️ Model Configuration Reference

| **Parameter**         | **Description**                                                 | **Tuning Impact**                                                      |
| --------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| **`device`**          | Hardware backend (`"cuda"` for GPU, `"cpu"` otherwise).         | Affects only speed, not behavior.                                      |
| **`vocab_size`**      | Number of unique tokens (symbols) the model can represent.      | Larger vocab = richer expressiveness, slightly higher compute cost.    |
| **`d_model`**         | Dimensionality of embeddings and hidden representations.        | Higher = more representational power; increases memory & compute.      |
| **`d_state`**         | Size of the state vector in recurrent/SSM (Mamba-style) layers. | Larger = longer memory span; smaller = faster adaptation.              |
| **`n_layers`**        | Total number of processing blocks (TinyBlocks).                 | Deeper networks learn more hierarchy but risk overfitting small data.  |
| **`n_heads`**         | Number of attention heads in the attention layer.               | More heads = better multi-pattern focus, more computation.             |
| **`max_seq`**         | Maximum sequence length supported for rotary embeddings (RoPE). | Should match your longest expected sequence; minimal effect otherwise. |
| **`attn_layer_idx`**  | Index of the layer that uses attention instead of SSM.          | Determines where global focus is applied in the hierarchy.             |
| **`base_decay`**      | Controls how quickly internal states forget old information.    | Lower = longer memory; higher = faster adaptation.                     |
| **`learning_rate`**   | Optimizer step size for gradient updates.                       | Too high → instability; too low → slow convergence.                    |
| **`training_epochs`** | Number of full training passes over data.                       | More epochs improve convergence until saturation.                      |
| **`eval_interval`**   | Frequency (in epochs) of evaluation/generation checkpoints.     | Adjust based on how often you want progress logs.                      |
| **`batch_size`**      | Number of sequences processed per training step.                | Larger = smoother gradients; smaller = more adaptive, less stable.     |
| **`seq_len`**         | Length of training sequences (context window).                  | Longer = captures more dependencies; slower per step.                  |
| **`context_factor`**  | Scaling factor for contextual blending in state updates.        | <1 = favors short-term memory; >1 = enhances persistence.              |
| **`state_path`**      | Directory for saved hidden states between runs.                 | Reset or clear when architecture or config changes.                    |
