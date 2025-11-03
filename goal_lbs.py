# implementation based on LLM-Integrated Bayesian SSMs for Multimodel Time-Series Forecasting, vibe coded by Grok, not yet integrated into TinyMamba
# https://arxiv.org/pdf/2510.20952

# ==============================
#  GoalLBS: Goal-Conditioned State Space Reasoner
#  → Extends LBS with goal encoding + steering
# ==============================

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Optional, List


# ------------------------------
# 1. Config
# ------------------------------
class CFG:
    latent_dim = 16
    gru_layers = 1
    sum_tokens = 8                # K <SUM> tokens
    prefix_tokens = 8             # projected state tokens
    lora_r = 8
    lora_alpha = 16
    kl_free_nats = 2.5
    alpha_val = 1.0
    alpha_text = 0.1
    alpha_kl = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()

# ------------------------------
# 2. LBS core model
# ------------------------------
class LBS(nn.Module):
    def __init__(self, llm, tokenizer):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.device = cfg.device

        # SSM (GRU)
        self.gru = nn.GRU(cfg.latent_dim, cfg.latent_dim,
                          num_layers=cfg.gru_layers, batch_first=True)

        # MLPs
        self.mlp_prior = nn.Linear(cfg.latent_dim, cfg.latent_dim * 2)   # mu, logvar
        self.mlp_post = nn.Linear(cfg.latent_dim*2 + 1, cfg.latent_dim * 2)  # h, y, s
        self.mlp_val = nn.Linear(cfg.latent_dim, 1)                     # reconstruct y
        self.proj_sum = nn.Linear(llm.config.hidden_size, cfg.latent_dim)   # summary vector
        self.proj_state = nn.Linear(cfg.latent_dim, cfg.prefix_tokens * llm.config.hidden_size)

        # token ids
        self.sum_ids = torch.tensor(
            [tokenizer.convert_tokens_to_ids(f"<SUM{i}>") for i in range(cfg.sum_tokens)],
            dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    # 2.1 Text → summary vector s_t  (compression)
    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> torch.Tensor:
        prompt = f"Encode the information into a sequence of vectors.\n{text}\n"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        # append K <SUM> tokens
        input_ids = torch.cat([input_ids, self.sum_ids.unsqueeze(0)], dim=1)

        with torch.no_grad():
            out = self.llm.model(input_ids=input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1]                     # (1, L, D)
        sum_hidden = hidden[0, -cfg.sum_tokens:]           # last K tokens
        s = self.proj_sum(sum_hidden)                      # (K, latent_dim)
        s = s.mean(dim=0, keepdim=True)                    # (1, latent_dim)
        return s

    # ------------------------------------------------------------------
    # 2.2 Prior → posterior (Kalman-style)
    # ------------------------------------------------------------------
    def filter_step(self,
                    x_prev: torch.Tensor,
                    h_prev: torch.Tensor,
                    y_t: torch.Tensor,
                    s_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: x̂_t (sample), prior_mu/var, post_mu/var
        """
        # prior
        _, h_t = self.gru(x_prev.unsqueeze(0), h_prev)   # h_t: (1,1,latent)
        prior = self.mlp_prior(h_t.squeeze(0))           # (1, 2*latent)
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)

        # posterior
        post_in = torch.cat([h_t.squeeze(0), y_t.unsqueeze(0), s_t], dim=-1)
        post = self.mlp_post(post_in)
        post_mu, post_logvar = post.chunk(2, dim=-1)

        # sample (reparameterize)
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std)
        x_t = post_mu + eps * std

        return x_t, (prior_mu, prior_logvar), (post_mu, post_logvar), h_t

    # ------------------------------------------------------------------
    # 2.3 Reconstruction loss (value)
    # ------------------------------------------------------------------
    def value_loss(self, x_t: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        y_pred = self.mlp_val(x_t)
        return F.mse_loss(y_pred.squeeze(-1), y_t)

    # ------------------------------------------------------------------
    # 2.4 Text generation loss (conditioned on state)
    # ------------------------------------------------------------------
    def text_loss(self, x_t: torch.Tensor, text: str) -> torch.Tensor:
        # project state → prefix embeddings
        prefix_emb = self.proj_state(x_t)                     # (1, prefix_tokens * D)
        prefix_emb = prefix_emb.view(1, cfg.prefix_tokens, -1)

        # build target ids
        prompt = f"Given this belief state, generate a textual forecast.\nDate: 2025-01-01\n"
        full_prompt = prompt + text
        target_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(self.device)

        # prepend prefix embeddings (no token ids)
        inputs_embeds = self.llm.model.model.embed_tokens(target_ids)
        inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)

        labels = target_ids.clone()
        labels = torch.cat([torch.full((1, cfg.prefix_tokens), -100, device=self.device), labels], dim=1)

        out = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        return out.loss

    # ------------------------------------------------------------------
    # 2.5 KL with free nats annealing
    # ------------------------------------------------------------------
    @staticmethod
    def kl_loss(prior_mu, prior_logvar, post_mu, post_logvar, step, total_steps):
        kl = -0.5 * torch.sum(1 + post_logvar - prior_logvar
                              - (post_mu - prior_mu).pow(2) / prior_logvar.exp()
                              - post_logvar.exp() / prior_logvar.exp())
        free_nats = cfg.kl_free_nats * max(0.0, 1.0 - step / total_steps)
        return torch.clamp(kl - free_nats, min=0.0)

    # ------------------------------------------------------------------
    # 2.6 Single training step (stateful)
    # ------------------------------------------------------------------
    def training_step(self,
                      y_t: torch.Tensor,
                      text: str,
                      x_prev: torch.Tensor,
                      h_prev: torch.Tensor,
                      step: int,
                      total_steps: int):
        s_t = self.encode_text(text)                         # (1, latent)

        x_t, prior, post, h_t = self.filter_step(x_prev, h_prev, y_t, s_t)

        L_val  = self.value_loss(x_t, y_t)
        L_text = self.text_loss(x_t.detach(), text)          # stop grad on state for stability
        L_kl   = self.kl_loss(prior[0], prior[1], post[0], post[1], step, total_steps)

        loss = cfg.alpha_val * L_val + cfg.alpha_text * L_text + cfg.alpha_kl * L_kl
        return loss, x_t.detach(), h_t.detach(), (L_val.item(), L_text.item(), L_kl.item())

# ------------------------------
# 3. GoalLBS: Goal-Conditioned LBS
# ------------------------------
class GoalLBS(nn.Module):
    def __init__(self, base_lbs: LBS, goal_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.lbs = base_lbs
        self.device = base_lbs.device
        self.latent_dim = base_lbs.gru.hidden_size

        # Goal encoder: text → target latent state
        if goal_encoder is None:
            self.goal_encoder = nn.Sequential(
                nn.Linear(base_lbs.llm.config.hidden_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.latent_dim),
                nn.Tanh()  # bound to reasonable range
            )
        else:
            self.goal_encoder = goal_encoder

        # Steering strength
        self.gamma = 0.5  # how hard to pull toward goal (tunable)

    # --------------------------------------------------------------
    # 1. Encode goal text → target latent state x_goal
    # --------------------------------------------------------------
    def encode_goal(self, goal_text: str) -> torch.Tensor:
        prompt = f"Goal: {goal_text}\nSummarize the desired future state."
        input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            hidden = self.lbs.llm.model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
        goal_emb = hidden[0, -1, :]  # last token
        x_goal = self.goal_encoder(goal_emb)
        return x_goal  # (latent_dim,)

    # --------------------------------------------------------------
    # 2. Steering: Modify prior to pull toward x_goal
    # --------------------------------------------------------------
    def steer_prior(self, prior_mu: torch.Tensor, x_goal: torch.Tensor) -> torch.Tensor:
        error = x_goal - prior_mu
        steered_mu = prior_mu + self.gamma * error
        return steered_mu

    # --------------------------------------------------------------
    # 3. Full forward: one step with goal steering
    # --------------------------------------------------------------
    def forward_step(self,
                     y_t: torch.Tensor,
                     text_t: str,
                     x_prev: torch.Tensor,
                     h_prev: torch.Tensor,
                     x_goal: torch.Tensor,
                     step: int,
                     total_steps: int) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns: x_t, h_t, losses
        """
        # 1. Text → summary
        s_t = self.lbs.encode_text(text_t)

        # 2. Prior from SSM
        _, h_t = self.lbs.gru(x_prev.unsqueeze(0), h_prev)
        prior = self.lbs.mlp_prior(h_t.squeeze(0))
        prior_mu, prior_logvar = prior.chunk(2, dim=-1)

        # 3. STEER PRIOR toward goal
        prior_mu = self.steer_prior(prior_mu, x_goal)

        # 4. Posterior (uses steered prior in KL)
        post_in = torch.cat([h_t.squeeze(0), y_t.unsqueeze(0), s_t], dim=-1)
        post = self.lbs.mlp_post(post_in)
        post_mu, post_logvar = post.chunk(2, dim=-1)

        # 5. Sample
        std = torch.exp(0.5 * post_logvar)
        eps = torch.randn_like(std)
        x_t = post_mu + eps * std

        # 6. Losses
        L_val = self.lbs.value_loss(x_t, y_t)
        L_text = self.lbs.text_loss(x_t.detach(), text_t)
        L_kl = self.lbs.kl_loss(prior_mu, prior_logvar, post_mu, post_logvar, step, total_steps)

        loss = (self.lbs.cfg.alpha_val * L_val +
                self.lbs.cfg.alpha_text * L_text +
                self.lbs.cfg.alpha_kl * L_kl)

        return x_t.detach(), h_t.detach(), {
            'loss': loss, 'val': L_val.item(), 'text': L_text.item(), 'kl': L_kl.item()
        }

    # --------------------------------------------------------------
    # 4. Generate Plan from Final State
    # --------------------------------------------------------------
    def generate_plan(self, x_final: torch.Tensor, steps_ahead: int = 5) -> str:
        plan = "Plan to achieve goal:\n"
        x = x_final.unsqueeze(0)
        h = torch.zeros(self.lbs.gru.num_layers, 1, self.latent_dim, device=self.device)

        for step in range(steps_ahead):
            # Project state → prefix
            prefix_emb = self.lbs.proj_state(x).view(1, self.lbs.cfg.prefix_tokens, -1)
            prompt = f"Given this belief state, generate the next step in the plan.\nDate: Step {step+1}\n"
            input_ids = self.lbs.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            inputs_embeds = self.lbs.llm.model.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([prefix_emb, inputs_embeds], dim=1)

            output = self.lbs.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.lbs.tokenizer.eos_token_id
            )
            step_text = self.lbs.tokenizer.decode(output[0].split(input_ids.shape[1] + self.lbs.cfg.prefix_tokens)[-1], skip_special_tokens=True)
            plan += f"Step {step+1}: {step_text}\n"

            # Roll forward (no obs)
            with torch.no_grad():
                _, h = self.lbs.gru(x.unsqueeze(0), h)
                prior = self.lbs.mlp_prior(h.squeeze(0))
                prior_mu, _ = prior.chunk(2, dim=-1)
                x = prior_mu  # deterministic roll

        return plan