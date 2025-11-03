# Config presets selector for model training vibe coded by GPT-5

import torch
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Core parameters
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size: int = 32
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 5
    n_heads: int = 8
    max_seq: int = 512
    attn_layer_idx: int = 2
    base_decay: float = 0.05
    learning_rate: float = 1e-3
    training_epochs: int = 200
    eval_interval: int = 20
    batch_size: int = 1
    seq_len: int = 20
    context_factor: float = 0.75
    state_path: str = "state"


# ---- 🔧 Preset Registry ---- #
_PRESETS = {
    "tiny": dict(
        d_model=32, d_state=8, n_layers=3, n_heads=4,
        seq_len=10, max_seq=128, base_decay=0.07,
        context_factor=0.6, learning_rate=2e-3
    ),
    "balanced": dict(
        d_model=64, d_state=16, n_layers=5, n_heads=8,
        seq_len=20, max_seq=512, base_decay=0.05,
        context_factor=0.75, learning_rate=1e-3
    ),
    "extended_memory": dict(
        d_model=96, d_state=32, n_layers=6, n_heads=8,
        seq_len=40, max_seq=1024, base_decay=0.03,
        context_factor=0.9, learning_rate=8e-4
    ),
    "high_context": dict(
        d_model=80, d_state=24, n_layers=6, n_heads=8,
        seq_len=30, max_seq=512, base_decay=0.04,
        context_factor=0.85, learning_rate=9e-4
    ),
}


# ---- 🚀 Loader ---- #
def get_config(preset: str = "balanced", **overrides) -> Config:
    """
    Returns a Config instance for the given preset.
    You can optionally override individual parameters like:
        cfg = get_config("tiny", learning_rate=5e-3, n_layers=4)
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(_PRESETS.keys())}")

    cfg_dict = dict(Config().__dict__)  # start from defaults
    cfg_dict.update(_PRESETS[preset])   # apply preset
    cfg_dict.update(overrides)          # apply user overrides

    cfg = Config(**cfg_dict)

    # Ensure the state path is unique to each preset (optional)
    cfg.state_path = os.path.join("state", preset)
    os.makedirs(cfg.state_path, exist_ok=True)

    print(f"[Config] Loaded preset: '{preset}' on device={cfg.device}")
    return cfg
