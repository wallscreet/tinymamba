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


# class TinyMambaLayer(nn.Module):
#     def __init__(self, d_model=16, d_state=16, decay_rate=0.05):
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.input_proj = nn.Linear(d_model, d_state)
#         self.state_proj = nn.Linear(d_state, d_state)
#         self.output_proj = nn.Linear(d_state, d_model)
#         self.gate = nn.Linear(d_model, d_state)

#     def forward(self, x, prev_state=None):
#         B, T, D = x.shape
#         state = self.input_proj(x)
#         if prev_state is not None:
#             decayed_state = prev_state * math.exp(-self.decay_rate)
#             state = state + self.state_proj(decayed_state).unsqueeze(1)
#         g = torch.sigmoid(self.gate(x))
#         state = state * g
#         out = self.output_proj(state)
#         return out, state[:, -1, :]


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
        B, T, _ = x.shape
        delta = self.in_proj(x)  # (B, T, d_state)

        if prev_state is not None:
            #print(f"prev_state shape: {prev_state.shape}") # debug shape = [8, 16]
            # s_prev = prev_state.unsqueeze(1).expand(-1, T, -1)
            if prev_state.shape[0] == x.shape[0]:
                assert prev_state.shape[-1] == self.d_state, (f"State dim mismatch: expected {self.d_state}, got {prev_state.shape[-1]}")
                s_prev = prev_state.unsqueeze(1).expand(-1, T, -1)
            else:
                s_prev = torch.zeros(B, T, self.d_state, device=x.device)
            #print(f"s_prev shape: {s_prev.shape}") # debug shape = [8, 3, 16]
            #print(f"s_proj shape: {self.s_proj.weight.shape}") # debug shape = [8, 8]
            s_trans = self.s_proj(s_prev)

            ctx = torch.cat([x, s_prev], dim=-1)
            decay_factor = self.decay_ctrl(ctx)
            decay = torch.exp(-self.base_decay * decay_factor)
            s_decayed = s_prev * decay

            h = delta + s_trans * s_decayed

            gate_in = torch.sigmoid(self.gate_x(x))
            gate_state = torch.sigmoid(self.gate_s(s_prev))
            h = h * gate_in * gate_state
        else:
            h = delta * torch.sigmoid(self.gate_x(x))

        out = self.out_proj(h)
        return out, h[:, -1, :]


class TinyAttentionLayer(nn.Module):
    def __init__(self, d_model=16, n_heads=2, max_seq=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.rope    = RoPE(self.d_head, max_seq=max_seq)

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Use standard scaled dot-product attention for simplicity
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

    def forward(self, x, prev_state=None, seq_idx=0):
        B, L, D = x.shape

        # Separate projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (B, H, L, d_head)
        q = q.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = k.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = v.view(B, L, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Apply RoPE to Q and K only
        q = self.rope.apply(q, offset=seq_idx)
        k = self.rope.apply(k, offset=seq_idx)

        # Back to (B, L, D)
        q = q.permute(0, 2, 1, 3).reshape(B, L, D)
        k = k.permute(0, 2, 1, 3).reshape(B, L, D)
        v = v.permute(0, 2, 1, 3).reshape(B, L, D)

        # Build causal mask
        mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()

        # MultiheadAttention expects (B, L, D)
        attn_out, _ = self.attn(q, k, v, attn_mask=mask, need_weights=False)
        return attn_out, None


class TinyBlock(nn.Module):
    def __init__(self, d_model, d_state=16, base_decay=0.05,
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
        x_norm = self.norm(x)

        if isinstance(self.layer, TinyAttentionLayer):
            out, new_state = self.layer(x_norm, prev_state, seq_idx=seq_idx)
        else:
            out, new_state = self.layer(x_norm, prev_state)
        return x + out, new_state


# SSM-Transformer hybrid model
class HybridTinyMambaLM(nn.Module):
    def __init__(self,
                 vocab_size=20,
                 d_model=16,
                 d_state=16,
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
                block = TinyBlock(d_model=d_model, d_state=d_state, base_decay=base_decay, layer_type='attn', n_heads=n_heads, max_seq=max_seq)
            else:
                block = TinyBlock(d_model=d_model, d_state=d_state, base_decay=base_decay, layer_type='gated_ssm')
            self.layers.append(block)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, prev_states=None, position_offset=0):
        x = self.embedding(input_ids)
        B, T = input_ids.shape
        if prev_states is None:
            prev_states = [None] * len(self.layers)

        new_states = []
        for i, (block, prev_state) in enumerate(zip(self.layers, prev_states)):
            if i == self.attn_layer_idx:
                out, state = block(x, prev_state, seq_idx=position_offset)
            else:
                out, state = block(x, prev_state)  # ← NO seq_idx
            x = out
            new_states.append(state)
        logits = self.lm_head(x)
        return logits, new_states

    def save_states(self, states, path="state"):
        os.makedirs(path, exist_ok=True)
        for i, s in enumerate(states):
            if s is not None:
                torch.save(s, os.path.join(path, f"layer_{i}.pt"))

    # def load_states(self, path="state", device="cpu"):
    #     states = []
    #     for i in range(len(self.layers)):
    #         fp = os.path.join(path, f"layer_{i}.pt")
    #         if os.path.exists(fp):
    #             s = torch.load(fp, map_location=device)
    #             # decay only SSM layers
    #             if s is not None and i != self.attn_layer_idx:
    #                 s = s * math.exp(-self.base_decay)
    #             states.append(s)
    #         else:
    #             states.append(None)
    #     return states
    def load_states(self, path="state", device="cpu"):
        os.makedirs(path, exist_ok=True)
        states = []

        for i, layer in enumerate(self.layers):
            fp = os.path.join(path, f"layer_{i}.pt")
            d_state = getattr(layer.layer, "d_state", self.d_state)  # default fallback

            if os.path.exists(fp):
                s = torch.load(fp, map_location=device)
                if s is not None and i != self.attn_layer_idx:
                    s = s * math.exp(-self.base_decay)
                if s is None:
                    s = torch.zeros(1, d_state, device=device)
            else:
                s = torch.zeros(1, d_state, device=device)
                print(f"[load_states] Missing layer_{i}.pt → initialized zeros ({d_state} dims)")

            states.append(s)

        return states



#=================== Train, Gen & REPL ===================
def train():
    vocab_size = 30
    model = HybridTinyMambaLM(
        vocab_size=vocab_size,
        d_model=32,
        d_state=16,
        n_layers=4,
        attn_layer_idx=1,
        base_decay=0.05
    )
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Random dummy data
    # inputs  = torch.randint(0, vocab_size, (8, 10), device=device)
    # targets = torch.randint(0, vocab_size, (8, 10), device=device)
    inputs  = torch.arange(0, 10).unsqueeze(0).repeat(8, 1) % vocab_size
    targets = (inputs + 1) % vocab_size
    
    # Load previous states if they exist
    prev_states = model.load_states("state", device=device)

    for epoch in range(42):
        optimizer.zero_grad()

        # Forward pass
        logits, prev_states = model(inputs, prev_states, position_offset=0)

        # Compute loss
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

        # Backprop
        loss.backward()
        optimizer.step()

        # Detach recurrent states from the computation graph
        if prev_states is not None:
            if isinstance(prev_states, (tuple, list)):
                prev_states = tuple(s.detach() for s in prev_states if s is not None)
            else:
                prev_states = prev_states.detach()

        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    # Persist final states
    model.save_states(prev_states)
    print("Training done, states persisted.")

def train_toy_pattern():
    vocab_size = 10
    seq_len = 6
    batch_size = 8

    model = HybridTinyMambaLM(
        vocab_size=vocab_size,
        d_model=32,
        d_state=16,
        n_layers=3,
        attn_layer_idx=1,
        base_decay=0.05
    )
    device = torch.device("cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # Simple modular increment pattern
    # Inputs:  [0, 1, 2, 3, 4, 5, 6, 7]
    # Targets: [1, 2, 3, 4, 5, 6, 7, 8]
    #inputs = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1) % vocab_size
    #targets = (inputs + 1) % vocab_size
    inputs  = torch.tensor([list(range(seq_len)) for _ in range(batch_size)], device=device)
    targets = torch.tensor([list(range(1, seq_len)) + [seq_len % vocab_size] for _ in range(batch_size)], device=device)

    prev_states = model.load_states("state", device=device)

    for epoch in range(420):
        optimizer.zero_grad()
        logits, prev_states = model(inputs, prev_states, position_offset=0)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        # Detach state to prevent graph reuse
        prev_states = [s.detach() if s is not None else torch.zeros(1, model.d_state, device=device)for s in prev_states]
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

    model.save_states(prev_states, path="state")
    print("Toy pattern training complete. States saved.")

def generate():
    vocab_size = 10
    seq_len    = 6
    model = HybridTinyMambaLM(
        vocab_size=vocab_size,
        d_model=16,
        d_state=16,
        n_layers=3,
        attn_layer_idx=1,
        base_decay=0.05
    )
    device = torch.device("cpu")
    model.to(device)

    prev_states = model.load_states(path="state", device=device)
    input_ids   = torch.tensor([[1, 2, 3, 4]], device=device)
    generated   = input_ids[0].tolist()
    pos         = input_ids.shape[1]  # next position to generate

    for _ in range(seq_len):
        logits, prev_states = model(input_ids, prev_states, position_offset=pos)
        probs = torch.softmax(logits[:, -1, :] / 1.0, dim=-1)  # temperature = 1.0
        next_tok = torch.multinomial(probs, num_samples=1)
        #next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        generated.append(next_tok.item())
        pos += 1

    model.save_states(prev_states, path="state")
    print("Generated sequence:", generated)
    print("States saved to ./state/")

def generate_pattern():
    vocab_size = 10
    model = HybridTinyMambaLM(
        vocab_size=vocab_size,
        d_model=32,
        d_state=16,
        n_layers=3,
        attn_layer_idx=1,
        base_decay=0.05
    )
    device = torch.device("cpu")
    model.to(device)

    prev_states = model.load_states(path="state", device=device)

    input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
    generated = input_ids[0].tolist()
    pos = input_ids.shape[1]

    for _ in range(10):  # generate 10 more tokens
        logits, prev_states = model(input_ids, prev_states, position_offset=pos)
        probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)  # temperature < 1 = more confident
        next_tok = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        generated.append(next_tok.item())
        pos += 1

    model.save_states(prev_states, path="state")
    print("Generated sequence:", generated)


class TinyMambaREPL(cmd.Cmd):
    intro = '\n🚀 Tiny Mamba! Type help or ? to list commands.\n'
    prompt = '(tinymamba) '

    def __init__(self, model_class):
        super().__init__()
        self.model_class = model_class

    def do_train(self, arg):
        'Run the training loop: train'
        #train_()
        train_toy_pattern()

    def do_generate(self, arg):
        'Run generation: generate'
        #generate()
        generate_pattern()

    def do_exit(self, arg):
        'Exit the REPL: exit'
        print("Goodbye! 👋")
        return True
    
    def do_quit(self, arg):
        'Quit the REPL: quit'
        return self.do_exit(arg)
    
    def do_reset(self, arg):
        'Delete saved states'
        import shutil
        if os.path.exists("state"):
            shutil.rmtree("state")
            print("States reset.")
        else:
            print("No states to reset.")

    # empty line to allow to repeat last command
    def emptyline(self):
        pass

def main():
    print("Tiny Mamba!")
    repl = TinyMambaREPL(HybridTinyMambaLM)
    repl.cmdloop()

if __name__ == "__main__":
    main()
