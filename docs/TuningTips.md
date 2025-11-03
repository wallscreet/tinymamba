# 🧭 Tuning Tips & Tradeoffs

These aren’t strict rules — think of them as a mental map while exploring your model’s behavior.

## 🧠 Model Capacity

| **Goal**                      | **Params to Adjust**                   | **Effect**                                             |
| ----------------------------- | -------------------------------------- | ------------------------------------------------------ |
| Capture more complex patterns | ↑ `d_model`, ↑ `n_layers`, ↑ `n_heads` | Improves expressivity but increases memory & compute.  |
| Simplify / run faster         | ↓ `d_model`, ↓ `n_layers`, ↓ `n_heads` | Reduces overfitting and speeds training, may underfit. |

## 🔁 Memory & Context

| **Goal**            | **Params to Adjust**                            | **Effect**                                         |
| ------------------- | ----------------------------------------------- | -------------------------------------------------- |
| Longer-term memory  | ↓ `base_decay`, ↑ `d_state`, ↑ `context_factor` | Retains information across longer spans.           |
| Faster adaptability | ↑ `base_decay`, ↓ `context_factor`              | Learns local or recent dependencies more strongly. |

## 🔍 Attention Placement

| **Goal**                            | **Params to Adjust**                    | **Effect**                                                |
| ----------------------------------- | --------------------------------------- | --------------------------------------------------------- |
| Global reasoning early              | `attn_layer_idx = 0`                    | Emphasizes broad relationships before deep abstraction.   |
| Local reasoning first, global later | `attn_layer_idx = middle or last layer` | Builds structured representation before global synthesis. |

## ⚡ Training Dynamics

| **Goal**                | **Params to Adjust**                       | **Effect**                                             |
| ----------------------- | ------------------------------------------ | ------------------------------------------------------ |
| Faster convergence      | Slightly ↑ `learning_rate`, ↓ `batch_size` | Learns quickly but may overshoot minima.               |
| Stable, smooth learning | ↓ `learning_rate`, ↑ `batch_size`          | Improves consistency, slower progress.                 |
| Frequent feedback       | ↓ `eval_interval`                          | More checkpoints and visibility, slower total runtime. |

## 🧩 Sequence Behavior

| **Goal**                    | **Params to Adjust**                | **Effect**                                  |
| --------------------------- | ----------------------------------- | ------------------------------------------- |
| Handle longer sequences     | ↑ `seq_len`, ↑ `max_seq`            | Better continuity, but higher memory usage. |
| Focus on local interactions | ↓ `seq_len`, lower `context_factor` | Encourages concise and reactive updates.    |

