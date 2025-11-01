import os
import torch
import torch.nn as nn
import math


# Rotary Positional Embeddings
class RoPE:
    def __init__(self, dim, max_seq=512):
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.cos_cached = torch.cos(freqs)
        self.sin_cached = torch.sin(freqs)

    def apply(self, x, seq_idx):
        cos = self.cos_cached[seq_idx, :].to(x.device)
        sin = self.sin_cached[seq_idx, :].to(x.device)
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)


# Context gated state space layer
class GatedTinyMambaLayer(nn.Module):
    def __init__(self, d_model=32, d_state=16, base_decay=0.05):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.base_decay = base_decay

        # Standard projections
        self.input_proj = nn.Linear(d_model, d_state)
        self.state_proj = nn.Linear(d_state, d_state)
        self.output_proj = nn.Linear(d_state, d_model)

        # Context-dependent gates
        self.input_gate = nn.Linear(d_model, d_state)
        self.state_gate = nn.Linear(d_state, d_state)
        self.context_gate = nn.Linear(d_state, d_state)

        # Adaptive decay controller
        self.decay_gate = nn.Sequential(
            nn.Linear(d_model + d_state, d_state),
            nn.Sigmoid()
        )

    def forward(self, x, prev_state=None):
        B, T, D = x.shape
        new_state = self.input_proj(x)

        if prev_state is not None:
            projected_state = self.state_proj(prev_state).unsqueeze(1)

            # Compute adaptive decay with context awareness
            context = torch.cat([x, projected_state.expand_as(x)], dim=-1)
            decay_factor = torch.exp(-self.base_decay * self.decay_gate(context))
            decayed_state = projected_state * decay_factor

            # Context-dependent gating
            input_gate = self.input_gate(x)
            state_gate = self.state_gate(projected_state)
            context_mix = torch.sigmoid(input_gate + state_gate + self.context_gate(x * projected_state))
            new_state = new_state * context_mix + decayed_state

        else:
            context_mix = torch.sigmoid(self.input_gate(x))
            new_state = new_state * context_mix

        out = self.output_proj(new_state)
        return out, new_state[:, -1, :]


# Tiny Mamba Layer
class TinyMambaLayer(nn.Module):
    def __init__(self, d_model=16, d_state=8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.input_proj = nn.Linear(d_model, d_state)
        self.state_proj = nn.Linear(d_state, d_state)
        self.output_proj = nn.Linear(d_state, d_model)
        self.gate = nn.Linear(d_model, d_state)

    def forward(self, x, prev_state=None):
        B, T, D = x.shape
        state = self.input_proj(x)
        if prev_state is not None:
            decayed_state = prev_state * math.exp(-self.decay_rate)
            state = state + self.state_proj(decayed_state).unsqueeze(1)
        g = torch.sigmoid(self.gate(x))
        state = state * g
        out = self.output_proj(state)
        return out, state[:, -1, :]


# Tiny Attention Layer
class TinyAttentionLayer(nn.Module):
    def __init__(self, d_model=16, n_heads=2, max_seq=512):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.rope = RoPE(d_model // n_heads, max_seq=max_seq)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x, prev_state=None, seq_idx=0):
        B, L, D = x.shape
        qkv = x.reshape(B, L, self.n_heads, self.d_head).permute(0,2,1,3)
        qkv = self.rope.apply(qkv, seq_idx).reshape(B, L, D)
        attn_out, _ = self.attn(qkv, qkv, qkv)
        return attn_out, None


# Hybrid Model
class HybridTinyMambaLM(nn.Module):
    def __init__(self, vocab_size=20, d_model=16, d_state=8, n_layers=3, attn_layer_idx=1, max_seq=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        self.attn_layer_idx = attn_layer_idx
        for i in range(n_layers):
            if i == attn_layer_idx:
                self.layers.append(TinyAttentionLayer(d_model=d_model, n_heads=2, max_seq=max_seq))
            else:
                self.layers.append(TinyMambaLayer(d_model=d_model, d_state=d_state))
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, prev_states=None, seq_start_idx=0):
        x = self.embedding(input_ids)
        new_states = []
        if prev_states is None:
            prev_states = [None] * len(self.layers)

        for i, (layer, prev_state) in enumerate(zip(self.layers, prev_states)):
            if i == self.attn_layer_idx:
                out, state = layer(x, prev_state, seq_idx=seq_start_idx)
            else:
                out, state = layer(x, prev_state)
            x = x + out
            new_states.append(state)
        logits = self.lm_head(x)
        return logits, new_states

    # Save/Load State
    def save_states(self, states, path="state"):
        os.makedirs(path, exist_ok=True)
        for i, s in enumerate(states):
            if s is not None:
                torch.save(s, os.path.join(path, f"layer_{i}.pt"))

    def load_states(self, path="state", device="cpu"):
        states = []
        for i in range(len(self.layers)):
            file_path = os.path.join(path, f"layer_{i}.pt")
            if os.path.exists(file_path):
                s = torch.load(file_path, map_location=device)
                # Apply decay to old state immediately when loading
                if s is not None:
                    s = s * math.exp(-self.decay_rate)
                states.append(s)
            else:
                states.append(None)
        return states


def train():
    # ---- Training Example ----
    vocab_size = 30
    model = HybridTinyMambaLM(vocab_size=vocab_size, d_model=32, d_state=16, n_layers=4, base_decay=0.05)
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Toy data (sequence modeling)
    inputs = torch.randint(0, vocab_size, (8, 10))
    targets = torch.randint(0, vocab_size, (8, 10))

    prev_states = model.load_states("state", device=device)

    for epoch in range(3):
        optimizer.zero_grad()
        logits, prev_states = model(inputs, prev_states)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    model.save_states(prev_states)
    print("Training done, states persisted.")


def generate():
    # Persistent Generation
    vocab_size = 10
    seq_len = 5
    model = HybridTinyMambaLM(vocab_size=vocab_size, d_model=16, d_state=8, n_layers=3)
    device = torch.device("cpu")

    # Load states from previous session (if any)
    prev_states = model.load_states(path="state", device=device)
    input_ids = torch.tensor([[1, 2, 3]])

    generated = []
    for t in range(seq_len):
        logits, prev_states = model(input_ids, prev_states, seq_start_idx=t)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated.append(next_token.item())
        input_ids = next_token.unsqueeze(0)

    # Save updated states for future sessions
    model.save_states(prev_states, path="state")

    print("Generated sequence:", generated)
    print("States saved to ./state/")


def main():
    print("Tiny Mamba!")


if __name__ == "__main__":
    main()
