#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_attention_dit import DiT, AlternateVLDiT



class ContinuousTimeEncoder(nn.Module):
    """Maps continuous time in seconds to sinusoidal embeddings."""

    def __init__(self, embedding_dim: int, max_period: float = 10000.0):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"`embedding_dim` must be > 0, got {embedding_dim}.")
        self.embedding_dim = int(embedding_dim)
        half_dim = max(1, self.embedding_dim // 2)
        denom = max(half_dim - 1, 1)
        freq_idx = torch.arange(half_dim, dtype=torch.float32)
        freqs = torch.exp(-math.log(float(max_period)) * freq_idx / float(denom))
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.to(dtype=torch.float32)
        angles = t.unsqueeze(-1) * self.freqs.to(device=t.device)
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if emb.shape[-1] < self.embedding_dim:
            emb = F.pad(emb, (0, self.embedding_dim - emb.shape[-1]))
        return emb[..., : self.embedding_dim]


def build_time_grid(
    horizon_sec: float,
    hz: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-sample natural-time grids t_i = i / hz and hz-derived token masks.

    Returns:
        t_grid: (B, T) float timestamps in seconds.
        time_valid: (B, T) bool mask from floor(horizon_sec * hz).
        n_tokens: (B,) long effective token count per sample.
    """
    if hz.ndim != 1:
        raise ValueError(f"`hz` must have shape [B], got {tuple(hz.shape)}.")
    if seq_len <= 0:
        raise ValueError(f"`seq_len` must be > 0, got {seq_len}.")
    if horizon_sec <= 0:
        raise ValueError(f"`horizon_sec` must be > 0, got {horizon_sec}.")

    device = hz.device
    bsz = hz.shape[0]
    hz_f = hz.to(device=device, dtype=torch.float32)
    if torch.any(hz_f <= 0):
        bad_idx = int(torch.nonzero(hz_f <= 0, as_tuple=False)[0].item())
        raise ValueError(f"`hz` must be > 0 for all samples, got hz[{bad_idx}]={float(hz_f[bad_idx])}.")

    n_tokens = torch.floor(float(horizon_sec) * hz_f).to(dtype=torch.long)
    if torch.any(n_tokens < 1):
        bad_idx = int(torch.nonzero(n_tokens < 1, as_tuple=False)[0].item())
        raise ValueError(
            f"Invalid effective token count from horizon_sec*hz: sample={bad_idx}, "
            f"horizon_sec={horizon_sec}, hz={float(hz_f[bad_idx])}, floor={int(n_tokens[bad_idx])}. "
            "Increase `horizon_sec` or ensure hz>=1."
        )
    n_tokens = torch.clamp(n_tokens, max=int(seq_len))
    token_idx = torch.arange(int(seq_len), dtype=torch.float32, device=device).unsqueeze(0).expand(bsz, -1)
    t_grid = token_idx / hz_f.unsqueeze(1)
    time_valid = token_idx.to(dtype=torch.long) < n_tokens.unsqueeze(1)
    return t_grid, time_valid, n_tokens

    


def swish(x):
    return x * torch.sigmoid(x)

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)

    def forward(self, actions: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        x = swish(self.W1(actions, cat_ids))
        return self.W2(x, cat_ids)

class VectorMLP(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 3:
            x = x.unsqueeze(1)
        return self.net(x)



        
"""
Conditional flow matching head.
The decoder predicts velocity directly and sampling integrates that velocity
from a noise state at t=0 to the action trajectory at t=1.
CFG is applied in velocity space along the linear path x_t=(1-t)z+tx.
"""


@dataclass
class ConditionalFlowMatchingConfig:
    # Action decoder output dimension and DiT capacity.
    action_dim: int = 7
    hidden_dim: int = 768
    num_layers: int = 12
    attention_heads: int = 12
    num_steps: int = 20
    cfg_drop_prob: float = 0.0
    # Inference controls.
    cfg_guidance_scale: float = 1.0
    num_inference_steps: int = 4
    num_timestep_buckets: int = 1000

    # Learned condition encoders project vision, VLM, and state streams into the DiT space.
    vlm_dim: int = 2048
    vision_dim: int = 768
    num_vision_tokens: int = 256
    num_target_vision_tokens: int = -1
    horizon_sec: float = 1.0
    use_state: bool = True
    state_dim: int = 8
    num_embodiments: int = 32
    # num_vision_queries: int = 64
    # qformer_layers: int = 2
    # enc_num_heads: int = 4
    # AlternateVLDiT splits cross-attention between image tokens and VLM tokens.
    interleave_self_attention: bool = True
    use_alternate_vldit: bool = False
    attend_text_every_n_blocks: int = 2
    # When enabled, action-time embeddings are passed into DiT instead of added outside.
    use_action_positional_embeddings: bool = True
    

    # Beta schedule for flow time sampling.
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    token_independent_noise: bool = False

    # --- Denoising-scheme comparison flags (all defaults keep official behavior) ---
    # One-step training: fix flow time tau for every sample/position (e.g. 0.0
    # trains pure noise -> action regression; inference uses num_inference_steps=1).
    fixed_train_tau: Optional[float] = None
    # FASTER-style Horizon-Aware Schedule (HAS). In LaWAM convention (tau=1 clean)
    # the warp is tau_i = clip(tau / (1 - u_i), 0, 1) with u_i = (1 - j^alpha) * u0,
    # j = i / (H_valid - 1): front positions finish denoising early, rear positions
    # use the full budget. `has_train_mix_prob` is the per-sample probability of
    # training on the warped (vs shared) tau; the same alpha/u0 drive inference.
    has_train_mix_prob: float = 0.0
    has_alpha: float = 1.0
    has_u0: float = 0.9
    # er50 early-readout auxiliary loss: velocity MSE reweighted by (1 - tau)^2,
    # the exact endpoint error of the one-jump readout x1 = x_tau + (1-tau) * v.
    early_readout_loss_weight: float = 0.0


class ConditionalFlowMatchingHead(nn.Module):
    def __init__(self, config: Optional[ConditionalFlowMatchingConfig] = None):
        super().__init__()
        self.config = config or ConditionalFlowMatchingConfig()
        self.action_horizon: Optional[int] = None

               
        # self.enc_h_t_t1 = LAMEncoder(
        #     context_dim=self.config.vision_dim,
        #     query_dim=self.config.hidden_dim,
        #     num_queries=self.config.num_vision_queries,
        #     num_layers=self.config.qformer_layers,
        # )
        self.enc_vlm = VectorMLP(in_dim=self.config.vlm_dim, hidden_dim=self.config.vision_dim)
        # self.enc_a_p_to_a = VectorMLP(in_dim=2 * self.config.hidden_dim, hidden_dim=self.config.hidden_dim)
        if self.config.use_state:
            self.enc_state = MultiEmbodimentActionEncoder(
                action_dim=self.config.state_dim,
                hidden_size=self.config.hidden_dim,
                num_embodiments=self.config.num_embodiments,
            )
        else:
            self.enc_state = None
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.config.action_dim,
            hidden_size=self.config.hidden_dim,
            num_embodiments=self.config.num_embodiments,
        )
        self.time_encoder = ContinuousTimeEncoder(embedding_dim=self.config.hidden_dim)
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.config.num_embodiments,
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.action_dim,
        )
        if int(self.config.num_target_vision_tokens) > 0:
            self.future_tokens: Optional[nn.Embedding] = nn.Embedding(
                self.config.num_target_vision_tokens, self.config.hidden_dim
            )
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        else:
            self.future_tokens = None

        # cross_attention_dim must be set so condition tokens are consumed by cross-attention blocks.
        DiTClass = AlternateVLDiT if self.config.use_alternate_vldit else DiT
        
        dit_kwargs = {
            "num_attention_heads": self.config.attention_heads,
            "attention_head_dim": int(self.config.hidden_dim // self.config.attention_heads),
            "output_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "interleave_self_attention": self.config.interleave_self_attention,
            "cross_attention_dim": self.config.vision_dim,
        }
        if self.config.use_action_positional_embeddings:
            dit_kwargs["positional_embeddings"] = "continuous_time"
        
        if self.config.use_alternate_vldit:
            dit_kwargs["attend_text_every_n_blocks"] = self.config.attend_text_every_n_blocks
        
        self.DiT = DiTClass(**dit_kwargs)
        self.cfg_embeddings = nn.Parameter(torch.randn(1, self.config.num_vision_tokens, self.config.vision_dim))
        
        self.beta_dist = torch.distributions.Beta(
            concentration1=self.config.noise_beta_alpha,
            concentration0=self.config.noise_beta_beta,
        )

    def _compute_dtype(self) -> torch.dtype:
        return self.action_encoder.W1.W.dtype

    def _expand_future_tokens(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.future_tokens is None or int(self.config.num_target_vision_tokens) <= 0:
            return None, None
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        if future_tokens.device != device or future_tokens.dtype != dtype:
            future_tokens = future_tokens.to(device=device, dtype=dtype)
        future_token_valid = torch.ones(
            (batch_size, future_tokens.shape[1]), dtype=torch.bool, device=device
        )
        return future_tokens, future_token_valid

    @staticmethod
    def _cast_if_needed(x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        return x if x.dtype == target_dtype else x.to(dtype=target_dtype)

    def _build_hidden_positional_embeddings(
        self,
        *,
        action_time_emb: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        has_state_token: bool,
        future_token_count: int,
    ) -> torch.Tensor:
        prefix_len = future_token_count + (1 if has_state_token else 0)
        if prefix_len == 0:
            return action_time_emb.to(device=device, dtype=dtype)
        prefix = torch.zeros(
            (batch_size, prefix_len, action_time_emb.shape[-1]),
            device=device,
            dtype=dtype,
        )
        return torch.cat((prefix, action_time_emb.to(device=device, dtype=dtype)), dim=1)

    @staticmethod
    def _build_hidden_timesteps(
        *,
        action_timesteps: torch.Tensor,
        token_valid: torch.Tensor,
        has_state_token: bool,
        future_token_count: int,
    ) -> torch.Tensor:
        prefix_len = future_token_count + (1 if has_state_token else 0)
        if prefix_len == 0:
            return action_timesteps
        valid = token_valid.to(device=action_timesteps.device, dtype=torch.float32)
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        prefix_timestep = (
            (action_timesteps.to(dtype=torch.float32) * valid).sum(dim=1, keepdim=True) / denom
        ).round().to(dtype=action_timesteps.dtype)
        prefix_timesteps = prefix_timestep.expand(-1, prefix_len)
        return torch.cat((prefix_timesteps, action_timesteps), dim=1)

    def _prepare_state_condition(
        self,
        *,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        embodiment_id: torch.Tensor,
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.config.use_state:
            return None
        if self.enc_state is None:
            raise ValueError("`enc_state` is None while `use_state=True`.")
        if state.shape != state_mask.shape:
            raise ValueError(
                f"`state` and `state_mask` must have the same shape, got "
                f"state={tuple(state.shape)}, state_mask={tuple(state_mask.shape)}."
            )
        state = self._cast_if_needed(state, model_dtype)
        state_mask_f = state_mask.to(device=state.device, dtype=model_dtype)
        masked_state = state * state_mask_f
        if masked_state.dim() == 2:
            masked_state = masked_state.unsqueeze(1)
        return self.enc_state(masked_state, embodiment_id)

    def sample_noise(
        self, shape: Tuple[int, ...], device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Align training/inference sampling with Isaac-GR00T gr00t_n1d6: z ~ N(0, I).
        return torch.randn(size=shape, dtype=dtype, device=device)

    def sample_time(self, bsize: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Sample diffusion time tau aligned with Isaac-GR00T gr00t_n1d6:
        tau = (1 - Beta(alpha, beta)) * noise_s.

        Keep this as the single tau schedule for this head. Do not mix it with
        the UniVLA/legacy variant `(noise_s - sample) / noise_s`.
        """
        # Sample in fp32 for numerical stability, then cast back for bf16/half training.
        sample = self.beta_dist.sample([bsize]).to(device=device, dtype=torch.float32)
        sample = (1.0 - sample) * float(self.config.noise_s)
        return sample.to(dtype=dtype)

    def _has_warp_tau(self, time: torch.Tensor, token_valid: torch.Tensor) -> torch.Tensor:
        """FASTER Horizon-Aware Schedule in LaWAM tau convention (tau=1 clean).

        time: [B, 1, 1] shared tau; token_valid: [B, T] bool. Returns [B, T, 1]:
        tau_i = clip(tau / (1 - u_i), 0, 1), u_i = (1 - j^alpha) * u0. Position 0
        (u=u0) reaches tau=1 once the global tau passes 1-u0; the last valid
        position (u=0) keeps the unwarped tau.
        """
        seq_len = int(token_valid.shape[1])
        h_eff = token_valid.to(torch.float32).sum(dim=1).clamp_min(2.0)  # [B]
        idx = torch.arange(seq_len, device=time.device, dtype=torch.float32)[None, :]
        j = (idx / (h_eff[:, None] - 1.0)).clamp(0.0, 1.0)
        u = (1.0 - j ** float(self.config.has_alpha)) * float(self.config.has_u0)  # [B, T]
        warped = (time.squeeze(-1) / (1.0 - u).clamp_min(1e-6)).clamp(0.0, 1.0)
        return warped.unsqueeze(-1).to(dtype=time.dtype)

    def _has_time_schedule(
        self, num_steps: int, valid_horizon: int, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute the HAS inference schedule.

        Returns (tau_sched [S+1, T], dt_sched [S, T]) over the padded horizon;
        positions beyond `valid_horizon` follow the last valid column (masked out
        downstream anyway). Front positions take one big first step and then sit
        clean (dt=0); the last position integrates uniformly over all S steps.
        """
        idx = torch.arange(seq_len, device=device, dtype=torch.float32)
        j = (idx / float(max(valid_horizon - 1, 1))).clamp(0.0, 1.0)
        u = (1.0 - j ** float(self.config.has_alpha)) * float(self.config.has_u0)  # [T]
        grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # global tau 0 -> 1
        tau_sched = (grid[:, None] / (1.0 - u)[None, :].clamp_min(1e-6)).clamp(0.0, 1.0)
        dt_sched = tau_sched[1:] - tau_sched[:-1]
        return tau_sched, dt_sched

    
    def forward(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        state: torch.Tensor, # [B, D]
        actions: torch.Tensor, # [B, T, K]
        action_hz: torch.Tensor,  # [B]
        embodiment_id: torch.Tensor,  # [B]
        state_mask: torch.Tensor,  # [B, D]
        actions_mask: torch.Tensor,  # [B, T, K]
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model_dtype = self._compute_dtype()
        h_t = self._cast_if_needed(h_t, model_dtype)
        h_t1_star = self._cast_if_needed(h_t1_star, model_dtype)
        h_vlm = self._cast_if_needed(h_vlm, model_dtype)
        actions = self._cast_if_needed(actions, model_dtype)
        device = actions.device
        batch_size = h_t.shape[0]

        actions_mask_f = actions_mask.to(device=device, dtype=model_dtype)
        action_hz_f = action_hz.to(device=device, dtype=torch.float32)
        if action_hz_f.ndim != 1 or action_hz_f.shape[0] != batch_size:
            raise ValueError(
                f"`action_hz` must have shape [B], got {tuple(action_hz_f.shape)} for batch_size={batch_size}."
            )
        data_token_valid = actions_mask.to(device=device, dtype=torch.bool).any(dim=-1)
        
        # Either draw one flow time per action token or one shared time per trajectory.
        if self.config.token_independent_noise:
            flat_token_count = int(actions.shape[0] * actions.shape[1])
            noise = self.sample_noise(
                (flat_token_count, actions.shape[-1]), device, actions.dtype
            ).view_as(actions)
            time = self.sample_time(flat_token_count, device, actions.dtype).view(
                actions.shape[0], actions.shape[1], 1
            )
        else:
            noise = self.sample_noise(actions.shape, device, actions.dtype)
            time = self.sample_time(actions.shape[0], device, actions.dtype)
            time = time[:, None, None]

        # One-step training: every sample/position trains at a fixed tau.
        fixed_tau = getattr(self.config, "fixed_train_tau", None)
        if fixed_tau is not None:
            time = torch.full_like(time, float(fixed_tau))

        # FASTER-style HAS: with prob has_train_mix_prob, warp the shared tau into
        # the per-position inference schedule (front near-clean, rear noisy).
        has_prob = float(getattr(self.config, "has_train_mix_prob", 0.0))
        if has_prob > 0.0 and time.shape[1] == 1:
            warped = self._has_warp_tau(time, data_token_valid)  # [B, T, 1]
            use_has = torch.rand(batch_size, 1, 1, device=device) < has_prob
            time = torch.where(use_has, warped, time.expand_as(warped))

        # Flow matching path: x_t=(1-t)z+tx, velocity target is constant along the path.
        x_t = (1 - time) * noise + time * actions
        velocity_target = actions - noise
        # Clamp bucket ids before passing them to the DiT timestep embedding.
        t_discretized = (time.squeeze(-1) * self.config.num_timestep_buckets).long()
        t_discretized = torch.clamp(t_discretized, 0, self.config.num_timestep_buckets - 1)
        if t_discretized.shape[1] == 1:
            t_discretized = t_discretized[:, 0]

        # The time grid determines which padded action tokens are valid for each action_hz.
        t_grid, hz_token_valid, _ = build_time_grid(
            horizon_sec=float(self.config.horizon_sec),
            hz=action_hz_f,
            seq_len=int(actions.shape[1]),
        )
        expected_total = int(hz_token_valid.sum().item())
        actual_total = int(data_token_valid.sum().item())
        if actual_total != expected_total:
            raise ValueError(
                "Action mask/time-grid mismatch in training (batch-level count): "
                f"data_total={actual_total}, hz_total={expected_total}, "
                f"horizon_sec={self.config.horizon_sec}. "
                "Please ensure dataloader action padding matches floor(horizon_sec * action_hz)."
            )
        token_valid = data_token_valid

        # Encode the noisy trajectory x_t; this is the action-token input to DiT.
        noisy_trajectory_emb = self.action_encoder(x_t, embodiment_id)
        time_emb = self.time_encoder(t_grid).to(dtype=noisy_trajectory_emb.dtype)
        action_time_emb = time_emb * token_valid.unsqueeze(-1).to(dtype=noisy_trajectory_emb.dtype)
        # if not self.config.use_action_positional_embeddings:
        #     noisy_trajectory_emb = noisy_trajectory_emb + action_time_emb
        noisy_trajectory_emb = noisy_trajectory_emb * token_valid.unsqueeze(-1).to(dtype=noisy_trajectory_emb.dtype)

        cond_state = self._prepare_state_condition(
            state=state,
            state_mask=state_mask,
            embodiment_id=embodiment_id,
            model_dtype=model_dtype,
        )
        cond_vlm = self.enc_vlm(h_vlm)  # [B, seq_len, vision_dim]

        bsz = h_t.shape[0]
        cfg_future = self.cfg_embeddings.expand(bsz, -1, -1)
        if self.training and self.config.cfg_drop_prob > 0.0:
            mask = (torch.rand(bsz, device=device) < self.config.cfg_drop_prob).view(bsz, 1, 1)
            cond_future = torch.where(mask, cfg_future, h_t1_star)
        else:
            cond_future = h_t1_star
        if self.training:
            # Keep cfg_embeddings in the autograd graph every step so DDP can
            # safely run with find_unused_parameters=False when cfg_drop_prob=0.
            cond_future = cond_future + (cfg_future * 0.0)

        # Conditions are concatenated as current vision, future/CFG vision, then VLM tokens.
        encoder_hidden_states = torch.cat((h_t, cond_future, cond_vlm), dim=1)
        future_tokens, future_token_valid = self._expand_future_tokens(
            batch_size=batch_size,
            device=device,
            dtype=noisy_trajectory_emb.dtype,
        )
        future_token_count = 0 if future_tokens is None else int(future_tokens.shape[1])
        hidden_positional_embeddings = None
        if self.config.use_action_positional_embeddings:
            hidden_positional_embeddings = self._build_hidden_positional_embeddings(
                action_time_emb=action_time_emb,
                batch_size=batch_size,
                device=device,
                dtype=noisy_trajectory_emb.dtype,
                has_state_token=bool(self.config.use_state),
                future_token_count=future_token_count,
            )
        dit_timestep = t_discretized
        if t_discretized.dim() == 2:
            # Per-position timesteps (token-independent noise or HAS-warped tau).
            dit_timestep = self._build_hidden_timesteps(
                action_timesteps=t_discretized,
                token_valid=token_valid,
                has_state_token=bool(self.config.use_state),
                future_token_count=future_token_count,
            )
        if self.config.use_state:
            state_token_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            if future_tokens is not None and future_token_valid is not None:
                hidden_states = torch.cat((cond_state, future_tokens, noisy_trajectory_emb), dim=1)
                hidden_attention_mask = torch.cat(
                    [state_token_valid, future_token_valid, token_valid], dim=1
                )
            else:
                hidden_states = torch.cat((cond_state, noisy_trajectory_emb), dim=1)
                hidden_attention_mask = torch.cat([state_token_valid, token_valid], dim=1)
        else:
            if future_tokens is not None and future_token_valid is not None:
                hidden_states = torch.cat((future_tokens, noisy_trajectory_emb), dim=1)
                hidden_attention_mask = torch.cat([future_token_valid, token_valid], dim=1)
            else:
                hidden_states = noisy_trajectory_emb
                hidden_attention_mask = token_valid
        
        # Bool masks use True for visible tokens; VLM padding comes from the original attention mask.
        num_vision = h_t.shape[1] + cond_future.shape[1]  # 256 + 256 = 512
        num_vlm = cond_vlm.shape[1]
        if attention_mask is not None:
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)  # [B, 512 + vlm_seq_len]
        else:
            encoder_attention_mask = None
        
        if self.config.use_alternate_vldit:
            num_h_t = h_t.shape[1]
            num_h_t1 = cond_future.shape[1]
            num_vlm = cond_vlm.shape[1]
            
            # AlternateVLDiT uses these masks to switch cross-attention between image and VLM tokens.
            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)
            
            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)
            
            dit_output = self.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=dit_timestep,
                hidden_attention_mask=hidden_attention_mask,
                image_mask=image_mask,
                vlm_mask=vlm_mask,
                encoder_attention_mask=encoder_attention_mask,
                hidden_positional_embeddings=hidden_positional_embeddings,
            )
        else:
            dit_output = self.DiT(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=dit_timestep,
                hidden_attention_mask=hidden_attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                hidden_positional_embeddings=hidden_positional_embeddings,
            )
        
        # Decoder output is velocity directly, aligned with GR00T flow matching.
        pred_velocity_all = self.action_decoder(dit_output, embodiment_id)
        pred_velocity = pred_velocity_all[:, -actions.shape[1] :, :]
        loss_elem = F.mse_loss(pred_velocity, velocity_target, reduction="none")
        valid = actions_mask_f
        robot_valid = (embodiment_id.to(device=device, dtype=torch.long) != 0).to(dtype=model_dtype)
        valid = valid * robot_valid.view(-1, 1, 1)
        denom = valid.sum().clamp_min(1.0)
        losses = (loss_elem * valid).sum() / denom
        readout_weight = float(getattr(self.config, "early_readout_loss_weight", 0.0))
        if readout_weight > 0.0:
            # || x_tau + (1-tau) v_pred - x1 ||^2 == (1-tau)^2 || v_pred - v_target ||^2,
            # so the early-readout objective is the velocity loss reweighted toward low tau.
            # `time` is [B,1,1] (shared) or [B,T,1] (per-position); both broadcast.
            readout_w = (1.0 - time.to(dtype=loss_elem.dtype)) ** 2
            loss_readout = (loss_elem * readout_w * valid).sum() / denom
            losses = losses + readout_weight * loss_readout
        return losses

    @torch.inference_mode()
    def sample_actions_cfg(
        self,
        h_t: torch.Tensor,
        h_t1_star: torch.Tensor,
        h_vlm: torch.Tensor,
        state: torch.Tensor,
        state_mask: torch.Tensor,
        action_hz: torch.Tensor,  # [B]
        embodiment_id: torch.Tensor,  # [B]
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_padded: bool = False,
        return_early_readout: bool = False,
        time_schedule: str = "const",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        device = h_t.device
        model_dtype = self._compute_dtype()
        h_t = self._cast_if_needed(h_t, model_dtype)
        h_t1_star = self._cast_if_needed(h_t1_star, model_dtype)
        h_vlm = self._cast_if_needed(h_vlm, model_dtype)
        batch_size = h_t.shape[0]
        action_hz_f = action_hz.to(device=device, dtype=torch.float32)
        if action_hz_f.ndim != 1 or action_hz_f.shape[0] != batch_size:
            raise ValueError(
                f"`action_hz` must have shape [B], got {tuple(action_hz_f.shape)} for batch_size={batch_size}."
            )
        n_from_hz = torch.floor(float(self.config.horizon_sec) * action_hz_f).to(dtype=torch.long)
        if torch.any(n_from_hz < 1):
            bad_idx = int(torch.nonzero(n_from_hz < 1, as_tuple=False)[0].item())
            raise ValueError(
                f"Invalid effective token count from horizon_sec*hz: sample={bad_idx}, "
                f"horizon_sec={self.config.horizon_sec}, hz={float(action_hz_f[bad_idx])}, "
                f"floor={int(n_from_hz[bad_idx])}. Increase `horizon_sec` or ensure hz>=1."
            )
        base_horizon = int(n_from_hz.max().item())
        if self.action_horizon is None:
            raise ValueError("`ConditionalFlowMatchingHead.action_horizon` must be set from policy config before inference.")
        action_horizon = int(self.action_horizon)
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be > 0, got {action_horizon}.")
        if base_horizon > action_horizon:
            raise ValueError(
                f"Required horizon from hz ({base_horizon}) exceeds configured action_horizon ({action_horizon})."
            )
        output_horizon = action_horizon if return_padded else base_horizon
        t_grid, time_valid, _ = build_time_grid(
            horizon_sec=float(self.config.horizon_sec),
            hz=action_hz_f,
            seq_len=int(action_horizon),
        )
        if num_inference_steps is None:
            num_inference_steps = int(getattr(self.config, "num_inference_steps", self.config.num_steps))
        if cfg_scale is None:
            cfg_scale = float(self.config.cfg_guidance_scale)
        x_t = self.sample_noise(
            shape=(batch_size, action_horizon, self.config.action_dim),
            device=device,
            dtype=model_dtype,
        )
        x_t = x_t * time_valid.unsqueeze(-1).to(dtype=x_t.dtype)

        dt = 1.0 / float(num_inference_steps)

        if time_schedule not in ("const", "has"):
            raise ValueError(f"Invalid time_schedule: {time_schedule!r} (const|has).")
        if return_early_readout and time_schedule != "const":
            raise ValueError("return_early_readout requires time_schedule='const'.")
        has_tau_sched = has_dt_sched = None
        if time_schedule == "has":
            has_tau_sched, has_dt_sched = self._has_time_schedule(
                num_steps=int(num_inference_steps),
                valid_horizon=int(base_horizon),
                seq_len=int(action_horizon),
                device=device,
            )
        early_readout_per_step: list = []

        cond_vlm = self.enc_vlm(h_vlm)  # [B, seq_len, vision_dim]
        cond_state = self._prepare_state_condition(
            state=state,
            state_mask=state_mask,
            embodiment_id=embodiment_id,
            model_dtype=model_dtype,
        )

        cond_encoder_hidden = torch.cat((h_t, h_t1_star, cond_vlm), dim=1)
        future_tokens, future_token_valid = self._expand_future_tokens(
            batch_size=batch_size,
            device=device,
            dtype=model_dtype,
        )

        num_vision = h_t.shape[1] + h_t1_star.shape[1]  # 256 + 256 = 512
        if attention_mask is not None:
            vlm_mask_bool = attention_mask.to(device=device, dtype=torch.bool)
            vision_mask_bool = torch.ones(batch_size, num_vision, dtype=torch.bool, device=device)
            encoder_attention_mask = torch.cat([vision_mask_bool, vlm_mask_bool], dim=1)  # [B, 512 + vlm_seq_len]
        else:
            encoder_attention_mask = None

        use_cfg = cfg_scale is not None and cfg_scale != 1.0
        if use_cfg:
            uncond_encoder_hidden = torch.cat((
                h_t,
                self.cfg_embeddings.expand(batch_size, -1, -1),
                cond_vlm
            ), dim=1)

        if self.config.use_alternate_vldit:
            num_h_t = h_t.shape[1]
            num_h_t1 = h_t1_star.shape[1]
            num_vlm = cond_vlm.shape[1]

            image_mask = torch.cat([
                torch.ones(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.zeros(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)

            vlm_mask = torch.cat([
                torch.zeros(batch_size, num_h_t + num_h_t1, dtype=torch.bool, device=device),
                torch.ones(batch_size, num_vlm, dtype=torch.bool, device=device)
            ], dim=1)

        for step in range(num_inference_steps):
            future_token_count_ts = 0 if future_tokens is None else int(future_tokens.shape[1])
            if time_schedule == "has":
                # Per-position timesteps from the HAS schedule (finished positions
                # sit at the last bucket and receive dt=0 below).
                action_timesteps = (
                    (has_tau_sched[step] * self.config.num_timestep_buckets)
                    .long()
                    .clamp(0, self.config.num_timestep_buckets - 1)[None, :]
                    .expand(batch_size, -1)
                )
                timesteps_tensor = self._build_hidden_timesteps(
                    action_timesteps=action_timesteps,
                    token_valid=time_valid,
                    has_state_token=bool(self.config.use_state),
                    future_token_count=future_token_count_ts,
                )
            else:
                t_cont = step / float(num_inference_steps)
                t_discretized = int(t_cont * self.config.num_timestep_buckets)
                t_discretized = min(self.config.num_timestep_buckets - 1, max(0, t_discretized))

                timesteps_tensor = torch.full(
                    size=(batch_size,), fill_value=t_discretized, device=device, dtype=torch.long
                )
            action_features = self.action_encoder(x_t, embodiment_id)
            time_emb = self.time_encoder(t_grid).to(dtype=action_features.dtype)
            action_time_emb = time_emb * time_valid.unsqueeze(-1).to(dtype=action_features.dtype)
            # if not self.config.use_action_positional_embeddings:
            #     action_features = action_features + action_time_emb
            action_features = action_features * time_valid.unsqueeze(-1).to(dtype=action_features.dtype)
            future_token_count = 0 if future_tokens is None else int(future_tokens.shape[1])
            hidden_positional_embeddings = None
            if self.config.use_action_positional_embeddings:
                hidden_positional_embeddings = self._build_hidden_positional_embeddings(
                    action_time_emb=action_time_emb,
                    batch_size=batch_size,
                    device=device,
                    dtype=action_features.dtype,
                    has_state_token=bool(self.config.use_state),
                    future_token_count=future_token_count,
                )

            if self.config.use_state:
                state_token_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
                if future_tokens is not None and future_token_valid is not None:
                    hidden_states = torch.cat((cond_state, future_tokens, action_features), dim=1)
                    hidden_attention_mask = torch.cat(
                        [state_token_valid, future_token_valid, time_valid], dim=1
                    )
                else:
                    hidden_states = torch.cat((cond_state, action_features), dim=1)
                    hidden_attention_mask = torch.cat([state_token_valid, time_valid], dim=1)
            else:
                if future_tokens is not None and future_token_valid is not None:
                    hidden_states = torch.cat((future_tokens, action_features), dim=1)
                    hidden_attention_mask = torch.cat([future_token_valid, time_valid], dim=1)
                else:
                    hidden_states = action_features
                    hidden_attention_mask = time_valid

            if self.config.use_alternate_vldit:
                model_output_cond = self.DiT(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cond_encoder_hidden,
                    timestep=timesteps_tensor,
                    hidden_attention_mask=hidden_attention_mask,
                    image_mask=image_mask,
                    vlm_mask=vlm_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    hidden_positional_embeddings=hidden_positional_embeddings,
                )
                pred_velocity_cond_all = self.action_decoder(model_output_cond, embodiment_id)
                pred_velocity_cond = pred_velocity_cond_all[:, -action_horizon:, :]

                if use_cfg:
                    model_output_uncond = self.DiT(
                        hidden_states=hidden_states,
                        encoder_hidden_states=uncond_encoder_hidden,
                        timestep=timesteps_tensor,
                        hidden_attention_mask=hidden_attention_mask,
                        image_mask=image_mask,
                        vlm_mask=vlm_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        hidden_positional_embeddings=hidden_positional_embeddings,
                    )
                    pred_velocity_uncond_all = self.action_decoder(model_output_uncond, embodiment_id)
                    pred_velocity_uncond = pred_velocity_uncond_all[:, -action_horizon:, :]
                    pred_velocity = pred_velocity_uncond + cfg_scale * (
                        pred_velocity_cond - pred_velocity_uncond
                    )
                else:
                    pred_velocity = pred_velocity_cond
            else:
                model_output_cond = self.DiT(
                    hidden_states=hidden_states,
                    encoder_hidden_states=cond_encoder_hidden,
                    timestep=timesteps_tensor,
                    hidden_attention_mask=hidden_attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    hidden_positional_embeddings=hidden_positional_embeddings,
                )
                pred_velocity_cond_all = self.action_decoder(model_output_cond, embodiment_id)
                pred_velocity_cond = pred_velocity_cond_all[:, -action_horizon:, :]

                if use_cfg:
                    model_output_uncond = self.DiT(
                        hidden_states=hidden_states,
                        encoder_hidden_states=uncond_encoder_hidden,
                        timestep=timesteps_tensor,
                        hidden_attention_mask=hidden_attention_mask,
                        encoder_attention_mask=encoder_attention_mask,
                        hidden_positional_embeddings=hidden_positional_embeddings,
                    )
                    pred_velocity_uncond_all = self.action_decoder(model_output_uncond, embodiment_id)
                    pred_velocity_uncond = pred_velocity_uncond_all[:, -action_horizon:, :]
                    pred_velocity = pred_velocity_uncond + cfg_scale * (
                        pred_velocity_cond - pred_velocity_uncond
                    )
                else:
                    pred_velocity = pred_velocity_cond

            pred_velocity = pred_velocity * time_valid.unsqueeze(-1).to(dtype=pred_velocity.dtype)
            if time_schedule == "has":
                x_t = x_t + has_dt_sched[step][None, :, None].to(dtype=x_t.dtype) * pred_velocity
            else:
                x_t = x_t + dt * pred_velocity
            x_t = x_t * time_valid.unsqueeze(-1).to(dtype=x_t.dtype)

            if return_early_readout:
                # Ahead-of-time decode: freeze the current velocity and jump the
                # remaining flow time in one step. Pure side-channel readout; the
                # main denoising trajectory above is untouched.
                tau_after = float(step + 1) * dt
                x1_hat = x_t + (1.0 - tau_after) * pred_velocity
                x1_hat = x1_hat * time_valid.unsqueeze(-1).to(dtype=x1_hat.dtype)
                early_readout_per_step.append(x1_hat[:, :output_horizon, :])

        if return_early_readout:
            # Staircase assembly: split the output horizon into `num_inference_steps`
            # groups by execution order; group g is decoded from the readout of
            # iteration g (earlier groups exit earlier, later groups get more steps).
            num_groups = int(num_inference_steps)
            staircase = torch.empty_like(early_readout_per_step[-1])
            group_bounds = [
                (g * output_horizon) // num_groups for g in range(num_groups + 1)
            ]
            for g in range(num_groups):
                lo, hi = group_bounds[g], group_bounds[g + 1]
                if lo < hi:
                    staircase[:, lo:hi, :] = early_readout_per_step[g][:, lo:hi, :]
            readout = {
                "staircase_actions": staircase,
                "per_step_x1_pred": torch.stack(early_readout_per_step, dim=0),
                "group_bounds": torch.tensor(group_bounds, dtype=torch.long),
            }
            return x_t[:, :output_horizon, :], readout

        return x_t[:, :output_horizon, :]
