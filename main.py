import os 
import math 
import torch
import torch.nn as nn
import torch.nn.functional as F
import cmd


# Config
class Config:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 32
    d_model = 32
    d_state = 16
    n_layers = 4
    n_heads = 2
    max_seq = 512
    attn_layer_idx = 2          # middle layer gets attention
    base_decay = 0.05
    learning_rate = 1e-3
    training_epochs = 420
    eval_interval = 20
    batch_size = 1
    seq_len = 20
    context_factor = .75
    state_path = "state"


class RoPE(nn.Module):
    def __init__(self, dim, max_seq=Config.max_seq):
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
        offset = int(offset)
        cos = self.cos[offset:offset + L]               # (L, dim//2)
        sin = self.sin[offset:offset + L]
        cos = cos.unsqueeze(0).unsqueeze(0)             # (1,1,L,dim//2)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x1 = x[..., :dh//2]
        x2 = x[..., dh//2:]
        rotated = torch.cat([x1 * cos - x2 * sin,
                             x1 * sin + x2 * cos], dim=-1)
        return rotated


class TinyAttentionLayer(nn.Module):
    def __init__(self, d_model=Config.d_model, n_heads=Config.n_heads, max_seq=Config.max_seq):
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
        mask = torch.triu(torch.ones(L, L, device=x.device) * float('-inf'), diagonal=1)
        
        # MultiheadAttention expects (B, L, D)
        attn_out, _ = self.attn(q, k, v, attn_mask=mask)
        
        return attn_out, None


class GatedTinyMambaLayer(nn.Module):
    def __init__(self, d_model=Config.d_model, d_state=Config.d_state, base_decay=Config.base_decay):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.base_decay = base_decay

        # Core projections
        self.in_proj = nn.Linear(d_model, d_state, bias=False)   # Δ (input → state)
        self.x_proj = nn.Linear(d_model, d_state, bias=False)    # additional input contribution
        self.s_proj = nn.Linear(d_state, d_state, bias=False)    # state transition
        self.out_proj = nn.Linear(d_state, d_model, bias=False)  # state → output

        # Selective gates
        self.gate_x = nn.Linear(d_model, d_state)
        self.gate_s = nn.Linear(d_state, d_state)

        # Adaptive decay
        self.decay_ctrl = nn.Sequential(
            nn.Linear(d_model + d_state, d_state),
            nn.Sigmoid()
        )

        # Perturbation feedback
        self.perturb_proj = nn.Linear(2 * d_state, d_state)  # learns direction of correction
        self.perturb_gate = nn.Linear(d_state, d_state)      # controls perturbation strength

    def forward(self, x, prev_state=None, goal=None):
        B, T, _ = x.shape
        delta = self.in_proj(x)

        if prev_state is not None:
            s_prev = (
                prev_state.unsqueeze(1).expand(-1, T, -1)
                if prev_state.shape[0] == B
                else torch.zeros(B, T, self.d_state, device=x.device)
            )

            ctx = torch.cat([x, s_prev], dim=-1)
            decay = torch.exp(-self.base_decay * self.decay_ctrl(ctx).clamp(0.0, 1.0))
            s_decayed = s_prev * decay

            s_pred = s_decayed + delta + self.x_proj(x) + 0.5 * self.s_proj(s_prev)

            # Inject goal into perturbation logic
            if goal is not None:
                goal = goal.unsqueeze(1).expand(-1, T, -1)  # match temporal dims
                perturb_in = torch.cat([s_pred, s_prev, goal], dim=-1)
            else:
                perturb_in = torch.cat([s_pred, s_prev], dim=-1)

            perturb_vec = torch.tanh(self.perturb_proj(perturb_in))
            perturb_strength = torch.sigmoid(self.perturb_gate(s_prev))
            s_corrected = s_pred + perturb_strength * perturb_vec

            gate_in = torch.sigmoid(self.gate_x(x))
            gate_state = torch.sigmoid(self.gate_s(s_prev))
            h = s_corrected * gate_in * gate_state

        else:
            h = delta * torch.sigmoid(self.gate_x(x))

        out = self.out_proj(h)
        return out, h[:, -1, :]


# TinyBlock (with optional attention)
class TinyBlock(nn.Module):
    def __init__(self, d_model, d_state, base_decay, use_attention=False):
        super().__init__()
        self.layer = (
            TinyAttentionLayer(d_model=d_model, n_heads=Config.n_heads, max_seq=Config.max_seq) 
            if use_attention else GatedTinyMambaLayer(d_model, d_state, base_decay))
        self.norm = nn.LayerNorm(d_model)
        self.use_attention = use_attention

    def forward(self, x, state=None):
        x_norm = self.norm(x)
        if self.use_attention:
            attn_out, _ = self.layer(x_norm)
            return x + attn_out, state  # ← preserve state
        else:
            out, new_state = self.layer(x_norm, state)
            return x + out, new_state

# HybridTinyMambaLM
class HybridTinyMambaLM(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.d_state = config.d_state
        self.n_layers = config.n_layers
        self.attn_layer_idx = config.attn_layer_idx
        self.base_decay = config.base_decay
        self.state_path = config.state_path

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)

        self.layers = nn.ModuleList([
            TinyBlock(config.d_model, config.d_state, config.base_decay,
                      use_attention=(i == config.attn_layer_idx))
            for i in range(config.n_layers)
        ])

        self.output = nn.Linear(config.d_model, config.vocab_size)

    # State handling
    def load_states(self, path=None, device="cpu", batch_size=1):
        if path is None:
            path = self.state_path
        os.makedirs(path, exist_ok=True)

        states = []
        for i, layer in enumerate(self.layers):
            fp = os.path.join(path, f"layer_{i}.pt")
            d_state = getattr(layer.layer, "d_state", self.d_state)

            if os.path.exists(fp):
                s = torch.load(fp, map_location=device)
                if s is not None and not layer.use_attention:
                    s = s * math.exp(-self.base_decay)

                # Resize to correct batch size
                if s.shape[0] > batch_size:
                    s = s[:batch_size, :].clone()
                elif s.shape[0] < batch_size:
                    s = s.mean(dim=0, keepdim=True).expand(batch_size, -1).clone()
            else:
                s = torch.zeros(batch_size, d_state, device=device)
                print(f"[load_states] Missing layer_{i}.pt → zeros ({d_state})")

            states.append(s.to(device))
        return states

    def save_states(self, states, path=None):
        if path is None:
            path = self.state_path
        os.makedirs(path, exist_ok=True)
        for i, s in enumerate(states):
            torch.save(s, os.path.join(path, f"layer_{i}.pt"))

    # Forward
    def forward(self, input_ids, prev_states=None, position_offset=0):
        x = self.input_norm(self.embedding(input_ids))  # (B, T, D)
        new_states = []

        for i, block in enumerate(self.layers):
            prev_state = None if prev_states is None else prev_states[i]
            x, new_state = block(x, prev_state)
            new_states.append(new_state)

        logits = self.output(x)
        return logits, new_states


# Training utilities
def generate(model, prompt, steps=Config.seq_len):
    model.eval()
    input_ids = torch.tensor([prompt], device=Config.device)
    states = model.load_states(Config.state_path, device=Config.device, batch_size=Config.batch_size)

    with torch.no_grad():
        for _ in range(steps):
            logits, states = model(input_ids, states)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)

    print("Generated sequence:", input_ids.squeeze().tolist())
    return input_ids.squeeze().tolist()


def train():
    print("Using device:", Config.device)
    model = HybridTinyMambaLM(Config).to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Simple +1 pattern data
    # inputs = torch.randint(0, cfg.vocab_size - 1, (cfg.batch_size, cfg.seq_len), device=cfg.device)
    # targets = (inputs + 1) % cfg.vocab_size
    seq = torch.arange(Config.seq_len, device=Config.device) % Config.vocab_size
    inputs = seq.unsqueeze(0).repeat(Config.batch_size, 1)
    targets = torch.roll(inputs, -1, dims=1)

    prev_states = model.load_states(Config.state_path, device=Config.device, batch_size=Config.batch_size)

    for epoch in range(Config.training_epochs):
        optimizer.zero_grad()
        logits, new_states = model(inputs, prev_states)
        loss = criterion(logits.view(-1, Config.vocab_size), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Detach for next epoch
        prev_states = [s.detach().clone() if s is not None else None for s in new_states]
        
        if (epoch + 1) % Config.eval_interval == 0:
            print(f"Epoch {epoch+1}/{Config.training_epochs} | Loss: {loss.item():.4f}")
            model.save_states(prev_states)
            generate(model, [0, 1, 2, 3, 4, 5, 6], steps=Config.seq_len)
    
    return model
    

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
        generate(self.model_class(Config()).to(Config.device), [0, 1, 2, 3, 4], steps=10)

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
        if os.path.exists(Config.state_path):
            shutil.rmtree(Config.state_path)
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
