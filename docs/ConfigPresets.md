# ⚙️ Recommended Baseline Presets

Each preset below is designed around a tradeoff: speed vs. capacity, local vs. global reasoning, and short-term vs. long-term context memory.

---

## 🪶 **Tiny (Quick Iteration)**

**Use for:** Fast debugging, architecture testing, training loop verification.

| Param                                                                                     | Value |
| ----------------------------------------------------------------------------------------- | ----- |
| `d_model`                                                                                 | 32    |
| `d_state`                                                                                 | 8     |
| `n_layers`                                                                                | 3     |
| `n_heads`                                                                                 | 4     |
| `seq_len`                                                                                 | 10    |
| `max_seq`                                                                                 | 128   |
| `base_decay`                                                                              | 0.07  |
| `context_factor`                                                                          | 0.6   |
| `learning_rate`                                                                           | 2e-3  |
| **Effect:** Tiny but snappy — learns patterns fast, forgets quickly, ideal for iteration. |       |

---

## ⚖️ **Balanced (Default Stable Setup)**

**Use for:** Most small-to-medium experiments; good tradeoff of cost and stability.

| Param                                                                                     | Value |
| ----------------------------------------------------------------------------------------- | ----- |
| `d_model`                                                                                 | 64    |
| `d_state`                                                                                 | 16    |
| `n_layers`                                                                                | 5     |
| `n_heads`                                                                                 | 8     |
| `seq_len`                                                                                 | 20    |
| `max_seq`                                                                                 | 512   |
| `base_decay`                                                                              | 0.05  |
| `context_factor`                                                                          | 0.75  |
| `learning_rate`                                                                           | 1e-3  |
| **Effect:** Balanced context retention and update speed — predictable and steady learner. |       |

---

## 🧠 **Extended Memory (Slow but Deep)**

**Use for:** Longer sequences, goal conditioning, or recurrent perturbation experiments.

| Param                                                                                                                   | Value |
| ----------------------------------------------------------------------------------------------------------------------- | ----- |
| `d_model`                                                                                                               | 96    |
| `d_state`                                                                                                               | 32    |
| `n_layers`                                                                                                              | 6     |
| `n_heads`                                                                                                               | 8     |
| `seq_len`                                                                                                               | 40    |
| `max_seq`                                                                                                               | 1024  |
| `base_decay`                                                                                                            | 0.03  |
| `context_factor`                                                                                                        | 0.9   |
| `learning_rate`                                                                                                         | 8e-4  |
| **Effect:** Retains information much longer; stable memory traces; slower to adapt but more consistent long-term logic. |       |

---

## ⚡ **High Context (Adaptive Thinker)**

**Use for:** Sequences that require combining many local and global relationships, e.g., reasoning or structured prediction.

| Param                                                                                                      | Value |
| ---------------------------------------------------------------------------------------------------------- | ----- |
| `d_model`                                                                                                  | 80    |
| `d_state`                                                                                                  | 24    |
| `n_layers`                                                                                                 | 6     |
| `n_heads`                                                                                                  | 8     |
| `seq_len`                                                                                                  | 30    |
| `max_seq`                                                                                                  | 512   |
| `base_decay`                                                                                               | 0.04  |
| `context_factor`                                                                                           | 0.85  |
| `learning_rate`                                                                                            | 9e-4  |
| **Effect:** Strong context blending; adapts across multiple reasoning horizons with moderate compute cost. |       |
