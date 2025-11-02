import os
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import cmd


class RoPE(nn.Module):
    def __init__(self, dim, max_seq=512):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq).float()
        freqs = torch.outer(t, inv_freq)                # (max_seq, dim//2)
        self.register_buffer('cos', torch.cos(freqs))
        self.register_buffer('sin', torch.sin(freqs))

    def apply(self, x, offset=0):
        # x: (B, H, L, d_head)
        B, H, L, dh = x.shape
        cos = self.cos[offset:offset + L]               # (L, dim//2)
        sin = self.sin[offset:offset + L]
        cos = cos.unsqueeze(0).unsqueeze(0)             # (1,1,L,dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x1 = x[..., :dh//2]
        x2 = x[..., dh//2:]
        rotated = torch.cat([x1 * cos - x2 * sin,
                             x1 * sin + x2 * cos], dim=-1)
        return rotated


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


# Context gated state space layer
class GatedTinyMambaLayer(nn.Module):
    def __init__(self, d_model=32, d_state=16, base_decay=0.05):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.base_decay = base_decay

        # Projections
        self.in_proj = nn.Linear(d_model, d_state, bias=False)
        self.x_proj = nn.Linear(d_model, d_state, bias=False)   # Δ (input contribution)
        self.s_proj = nn.Linear(d_state, d_state, bias=False)   # B (state transition)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)

        # Selective gates (like Mamba's)
        self.gate_x = nn.Linear(d_model, d_state)   # Input gate
        self.gate_s = nn.Linear(d_state, d_state)   # State gate

        # Adaptive decay: context → scalar per state dim
        self.decay_ctrl = nn.Sequential(
            nn.Linear(d_model + d_state, d_state),
            nn.Sigmoid()  # 0 to 1 → multiply base_decay
        )

    def forward(self, x, prev_state=None):
        """
        x: (B, T, d_model)
        prev_state: (B, d_state) or None
        """
        B, T, _ = x.shape

        # Input contribution: Δ_t = in_proj(x)
        delta = self.in_proj(x)  # (B, T, d_state)

        if prev_state is not None:
            # Expand previous state to sequence length
            s_prev = prev_state.unsqueeze(1)                 # (B, 1, d_state)
            s_prev = s_prev.expand(-1, T, -1)                # (B, T, d_state)

            # State transition: B * prev_state
            s_trans = self.s_proj(s_prev)                    # (B, T, d_state)

            # Context-aware decay: base_decay * sigmoid(ctrl)
            # decay is intended to simulate temporal memory decay in humans (maybe?)
            ctx = torch.cat([x, s_prev], dim=-1)              # (B, T, d_model + d_state)
            decay_factor = self.decay_ctrl(ctx)               # (B, T, d_state) ∈ [0,1]
            decay = torch.exp(-self.base_decay * decay_factor)
            s_decayed = s_prev * decay                        # (B, T, d_state)

            # Combine: h_t = Δ_t + B * decayed_prev
            h = delta + s_trans * s_decayed

            # Gating: selective parameter style
            gate_in = torch.sigmoid(self.gate_x(x))           # (B, T, d_state)
            gate_state = torch.sigmoid(self.gate_s(s_prev))   # (B, T, d_state)
            gate = gate_in * gate_state                       # Elementwise selectivity
            h = h * gate

        else:
            # No prev state: just gated input
            h = delta
            gate_in = torch.sigmoid(self.gate_x(x))
            h = h * gate_in

        out = self.out_proj(h)                                # (B, T, d_model)
        new_state = h[:, -1, :]                               # (B, d_state)

        return out, new_state


class TinyAttentionLayer(nn.Module):
    def __init__(self, d_model=16, n_heads=2, max_seq=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.attn    = nn.MultiheadAttention(embed_dim=d_model,
                                             num_heads=n_heads,
                                             batch_first=True)
        self.rope    = RoPE(self.d_head, max_seq=max_seq)

    def forward(self, x, prev_state=None, seq_idx=0):
        B, L, D = x.shape

        # reshape to (B, H, L, d_head)
        qkv = x.reshape(B, L, self.n_heads, self.d_head) \
               .permute(0, 2, 1, 3)                            # (B,H,L,d_h)

        # RoPE on Q and K (V stays untouched)
        q = self.rope.apply(qkv, offset=seq_idx)
        k = self.rope.apply(qkv, offset=seq_idx)
        v = qkv

        # back to (B, L, D) for MultiheadAttention
        q = q.permute(0, 2, 1, 3).reshape(B, L, D)
        k = k.permute(0, 2, 1, 3).reshape(B, L, D)
        v = v.permute(0, 2, 1, 3).reshape(B, L, D)

        # causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'),
                          diagonal=1).bool()

        attn_out, _ = self.attn(q, k, v,
                                attn_mask=mask,
                                need_weights=False)
        return attn_out, None


class TinyBlock(nn.Module):
    def __init__(self, d_model, d_state=8, base_decay=0.05,
                 layer_type='gated_ssm', n_heads=2, max_seq=512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        if layer_type == 'gated_ssm':
            self.layer = GatedTinyMambaLayer(d_model=d_model,
                                             d_state=d_state,
                                             base_decay=base_decay)
        elif layer_type == 'attn':
            self.layer = TinyAttentionLayer(d_model=d_model,
                                            n_heads=n_heads,
                                            max_seq=max_seq)
        else:
            raise ValueError(f"Unknown layer_type {layer_type}")

    def forward(self, x, prev_state=None, seq_idx=0):
        x_norm = self.norm(x)                                   # pre-normalization
        out, new_state = self.layer(x_norm, prev_state, seq_idx)
        return x + out, new_state                               # residual


# SSM-Transformer hybrid model
class HybridTinyMambaLM(nn.Module):
    def __init__(self,
                 vocab_size=20,
                 d_model=16,
                 d_state=8,
                 n_layers=3,
                 attn_layer_idx=1,
                 max_seq=512,
                 base_decay=0.05,
                 n_heads=2):
        super().__init__()
        self.d_model     = d_model
        self.d_state     = d_state
        self.base_decay  = base_decay
        self.max_seq     = max_seq
        self.n_heads     = n_heads
        self.attn_layer_idx = attn_layer_idx

        self.embedding = nn.Embedding(vocab_size, d_model)

        # build layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == attn_layer_idx:
                block = TinyBlock(d_model=d_model,
                                  d_state=d_state,
                                  base_decay=base_decay,
                                  layer_type='attn',
                                  n_heads=n_heads,
                                  max_seq=max_seq)
            else:
                block = TinyBlock(d_model=d_model,
                                  d_state=d_state,
                                  base_decay=base_decay,
                                  layer_type='gated_ssm')
            self.layers.append(block)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, prev_states=None, seq_start_idx=0):
        x = self.embedding(input_ids)                               # (B, T, d_model)
        if prev_states is None:
            prev_states = [None] * len(self.layers)

        new_states = []
        pos = seq_start_idx
        for layer, prev_state in zip(self.layers, prev_states):
            # Attention layers need the *absolute* position offset
            out, state = layer(x, prev_state, seq_idx=pos)
            x = out
            new_states.append(state)
            # only advance position for the next token when we are in generation
            # (training uses a full sequence, so we keep pos=0)
            if input_ids.shape[1] == 1:                             # single-token step (generation)
                pos += 1
        logits = self.lm_head(x)
        return logits, new_states

    def save_states(self, states, path="state"):
        os.makedirs(path, exist_ok=True)
        for i, s in enumerate(states):
            if s is not None:
                torch.save(s, os.path.join(path, f"layer_{i}.pt"))

    def load_states(self, path="state", device="cpu"):
        states = []
        for i in range(len(self.layers)):
            fp = os.path.join(path, f"layer_{i}.pt")
            if os.path.exists(fp):
                s = torch.load(fp, map_location=device)
                # decay only SSM layers
                if s is not None and i != self.attn_layer_idx:
                    s = s * math.exp(-self.base_decay)
                states.append(s)
            else:
                states.append(None)
        return states


def train():
    vocab_size = 30
    model = HybridTinyMambaLM(vocab_size=vocab_size,
                              d_model=32,
                              d_state=16,
                              n_layers=4,
                              attn_layer_idx=1,
                              base_decay=0.05)
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    inputs  = torch.randint(0, vocab_size, (8, 10), device=device)
    targets = torch.randint(0, vocab_size, (8, 10), device=device)

    prev_states = model.load_states("state", device=device)

    for epoch in range(3):
        optimizer.zero_grad()
        logits, prev_states = model(inputs, prev_states, seq_start_idx=0)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    model.save_states(prev_states)
    print("Training done, states persisted.")


def generate():
    vocab_size = 10
    seq_len    = 5
    model = HybridTinyMambaLM(vocab_size=vocab_size,
                              d_model=16,
                              d_state=8,
                              n_layers=3,
                              attn_layer_idx=1,
                              base_decay=0.05)
    device = torch.device("cpu")
    model.to(device)

    prev_states = model.load_states(path="state", device=device)
    input_ids   = torch.tensor([[1, 2, 3]], device=device)         # prompt
    generated   = input_ids[0].tolist()
    pos         = input_ids.shape[1]                               # start after prompt

    for _ in range(seq_len):
        logits, prev_states = model(input_ids, prev_states, seq_start_idx=pos)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (1,1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        generated.append(next_tok.item())
        pos += 1

    model.save_states(prev_states, path="state")
    print("Generated sequence:", generated)
    print("States saved to ./state/")


class TinyMambaREPL(cmd.Cmd):
    intro = '\n🚀 Tiny Mamba! Type help or ? to list commands.\n'
    prompt = '(tinymamba) '

    def __init__(self, model_class):
        super().__init__()
        self.model_class = model_class

    def do_train(self, arg):
        'Run the training loop: train'
        train()

    def do_generate(self, arg):
        'Run generation: generate'
        generate()

    def do_quit(self, arg):
        'Quit the REPL: quit'
        return self.do_exit(arg)

    # empty line to allow to repeat last command
    def emptyline(self):
        pass

def main():
    print("Tiny Mamba!")
    repl = TinyMambaREPL(HybridTinyMambaLM)
    repl.cmdloop()

if __name__ == "__main__":
    main()
