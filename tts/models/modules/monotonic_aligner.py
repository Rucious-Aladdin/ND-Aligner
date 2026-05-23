from typing import override

import torch
import torch.nn as nn
from ..submodules.sequential_unet import UNet2d


class UNetUnaryPotentialPredictor(nn.Module):
    """
    U-Net unary potential predictor.

    Pairwise input:
        [
            f_s(h_spec_t);
            f_x(h_text_j);
            f_c(cond);
            progress_features(t, j)
        ]

    Progress features:
        t_pos        = t / (spec_lengths[b] - 1)
        j_pos        = j / (text_lengths[b] - 1)
        diag         = t_pos - j_pos
        length_ratio = spec_lengths[b] / text_lengths[b]

    Returns:
        unary_potential: raw pairwise unary score e[t, j], (B, T_s, T_t)
    """

    progress_dim: int = 4

    def __init__(
        self,
        dim_spec: int,
        dim_text: int,
        dim_latent: int,
        dim_cond: int,
        cond_channels: int = 32,
        base_dim: int = 16,
        groups: int = 8,
        unary_scale_init: float = -2.0,
    ):
        super().__init__()

        self.dim_latent = dim_latent
        self.cond_channels = cond_channels

        self.spec_proj = nn.Sequential(
            nn.Linear(dim_spec, dim_latent * 2),
            nn.LayerNorm(dim_latent * 2),
            nn.GELU(),
            nn.Linear(dim_latent * 2, dim_latent),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(dim_text, dim_latent * 2),
            nn.LayerNorm(dim_latent * 2),
            nn.GELU(),
            nn.Linear(dim_latent * 2, dim_latent),
        )

        self.cond_proj = nn.Linear(dim_cond, cond_channels)

        self.unet = UNet2d(
            dim_in=2 * dim_latent + cond_channels + self.progress_dim,
            base_dim=base_dim,
            groups=groups,
        )

        # Keeps early evidence small.
        self.evidence_scale = nn.Parameter(
            torch.tensor(float(unary_scale_init), dtype=torch.float32)
        )

    @staticmethod
    def _make_progress_features(
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        T_s: int,
        T_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns:
            progress: (B, T_s, T_t, 4)

        Channels:
            t_pos        = t / (spec_lengths[b] - 1)
            j_pos        = j / (text_lengths[b] - 1)
            diag         = t_pos - j_pos
            length_ratio = spec_lengths[b] / text_lengths[b]
        """
        B = spec_lengths.size(0)

        t = torch.arange(T_s, device=device, dtype=dtype).view(1, T_s, 1, 1)
        j = torch.arange(T_t, device=device, dtype=dtype).view(1, 1, T_t, 1)

        spec_len = spec_lengths.to(device=device, dtype=dtype).view(B, 1, 1, 1)
        text_len = text_lengths.to(device=device, dtype=dtype).view(B, 1, 1, 1)

        t_pos = t / (spec_len - 1.0).clamp_min(1.0)
        j_pos = j / (text_len - 1.0).clamp_min(1.0)

        diag = t_pos - j_pos

        length_ratio = (spec_len / text_len.clamp_min(1.0)).expand(B, T_s, T_t, 1)

        t_pos = t_pos.expand(B, T_s, T_t, 1)
        j_pos = j_pos.expand(B, T_s, T_t, 1)
        diag = diag.expand(B, T_s, T_t, 1)

        return torch.cat(
            [
                t_pos,
                j_pos,
                diag,
                length_ratio,
            ],
            dim=-1,
        )

    @override
    def forward(
        self,
        h_spec: torch.Tensor,  # (B, C_s, T_s)
        h_text: torch.Tensor,  # (B, C_t, T_t)
        cond: torch.Tensor,  # (B, D_cond)
        spec_lengths: torch.Tensor,  # (B,)
        text_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Returns:
            unary_potential: raw pairwise unary score e[t, j], (B, T_s, T_t)
        """
        h_spec_t = h_spec.transpose(1, 2).contiguous()  # (B, T_s, C_s)
        h_text_t = h_text.transpose(1, 2).contiguous()  # (B, T_t, C_t)

        h_spec_t = self.spec_proj(h_spec_t)
        h_text_t = self.text_proj(h_text_t)

        B, T_s, C = h_spec_t.shape
        _, T_t, _ = h_text_t.shape

        spec_pair = h_spec_t.unsqueeze(2).expand(B, T_s, T_t, C)
        text_pair = h_text_t.unsqueeze(1).expand(B, T_s, T_t, C)

        cond_feat = self.cond_proj(cond)
        cond_pair = cond_feat.view(B, 1, 1, self.cond_channels).expand(
            B,
            T_s,
            T_t,
            self.cond_channels,
        )

        progress_pair = self._make_progress_features(
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
            T_s=T_s,
            T_t=T_t,
            device=h_spec.device,
            dtype=h_spec.dtype,
        )

        pairwise = torch.cat(
            [
                spec_pair,
                text_pair,
                cond_pair,
                progress_pair,
            ],
            dim=-1,
        )  # (B, T_s, T_t, 2C + cond_channels + 4)

        pairwise = pairwise.permute(0, 3, 1, 2).contiguous()

        unary_potential = self.unet(pairwise).squeeze(1)
        unary_potential = unary_potential * self.evidence_scale.exp()

        return unary_potential


class MonotonicCRFAligner(nn.Module):
    """
    Unary-only monotonic latent-path CRF aligner.

    Path topology:
        z_0 = 0
        z_{T-1} = N-1
        z_t -> z_t      stay
        z_t -> z_t + 1  advance

    There is no learned transition score.
    Allowed stay/advance transitions have score 0.
    Disallowed transitions are excluded by the monotonic reachability mask.

    Path score:
        S(z) = sum_t log_b[t, z_t]

    where log_b is a locally normalized unary potential.
    """

    def __init__(
        self,
        dim_spec: int,
        dim_text: int,
        dim_cond: int,
        dim_unary_latent: int,
        cond_channels: int = 32,
        unet_base_dim: int = 16,
        unet_groups: int = 8,
        unary_support_type: str = "local",  # or "global"
        unary_radius: int = 10,
        unary_temperature: float = 1.0,
        unary_scale_init: float = -2.0,
    ):
        super().__init__()

        self.dim_spec = dim_spec
        self.dim_text = dim_text

        if unary_support_type not in ("local", "global"):
            raise ValueError(
                f"unary_support_type must be either 'local' or 'global', got {unary_support_type!r}."
            )

        self.unary_support_type = unary_support_type
        self.unary_local_radius = int(unary_radius)
        self.unary_temperature = float(unary_temperature)

        self.unary_predictor = UNetUnaryPotentialPredictor(
            dim_spec=dim_spec,
            dim_text=dim_text,
            dim_latent=dim_unary_latent,
            dim_cond=dim_cond,
            cond_channels=cond_channels,
            base_dim=unet_base_dim,
            groups=unet_groups,
            unary_scale_init=unary_scale_init,
        )

    @override
    def forward(
        self,
        h_spec: torch.Tensor,  # (B, C_s, T_s)
        spec_mask: torch.Tensor,  # (B, T_s) or (B, 1, T_s)
        h_text: torch.Tensor,  # (B, C_t, T_t)
        text_mask: torch.Tensor,  # (B, T_t) or (B, 1, T_t)
        cond: torch.Tensor,  # (B, D_cond)
        return_hard: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if spec_mask.dim() == 3:
            spec_mask = spec_mask.squeeze(1)
        if text_mask.dim() == 3:
            text_mask = text_mask.squeeze(1)

        spec_mask = spec_mask.bool()
        text_mask = text_mask.bool()

        spec_lengths = spec_mask.sum(dim=-1).long()
        text_lengths = text_mask.sum(dim=-1).long()

        if torch.any(spec_lengths < text_lengths):
            raise ValueError(
                "Strict-monotone positive-duration alignment requires "
                + "spec_lengths[b] >= text_lengths[b] for every batch item."
            )

        spec_mask_f = spec_mask.unsqueeze(1).to(dtype=h_spec.dtype)
        text_mask_f = text_mask.unsqueeze(1).to(dtype=h_text.dtype)

        h_spec = h_spec * spec_mask_f
        h_text = h_text * text_mask_f

        raw_unary = self.unary_predictor(
            h_spec=h_spec,
            h_text=h_text,
            cond=cond,
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
        )  # (B, T_s, T_t)

        _, T_speech, T_text = raw_unary.shape
        device = raw_unary.device

        length_valid = spec_mask.unsqueeze(2) & text_mask.unsqueeze(1)

        reachable = self._strict_reachability_mask(
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
            T_speech=T_speech,
            T_text=T_text,
            device=device,
        )

        state_valid = length_valid & reachable

        masked_raw_unary = raw_unary.masked_fill(
            ~state_valid,
            self.neg_large,
        )

        (
            gamma,
            log_alpha,
            log_beta,
            log_gamma,
            raw_log_z,
            log_b,
        ) = self._forward_backward(
            raw_unary=raw_unary,
            state_valid=state_valid,
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
        )

        gamma = gamma.masked_fill(~state_valid, 0.0)
        durations = gamma.sum(dim=1)

        norm_log_z = raw_log_z / spec_lengths.float()

        base_outputs = (
            gamma,
            log_alpha,
            log_beta,
            log_gamma,
            raw_log_z,
            norm_log_z,
            raw_unary,
            masked_raw_unary,
            log_b,
            durations,
        )

        if not return_hard:
            return base_outputs

        hard_gamma, hard_durations, viterbi_path, viterbi_logp = self._viterbi_decode(
            raw_unary=raw_unary,
            state_valid=state_valid,
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
        )

        return (
            *base_outputs,
            hard_gamma,
            hard_durations,
            viterbi_path,
            viterbi_logp,
        )

    def update_temperature(self, new_temperature: float) -> None:
        if new_temperature <= 0.0:
            raise ValueError("Temperature must be positive.")
        self.unary_temperature = float(new_temperature)

    def _forward_backward(
        self,
        raw_unary: torch.Tensor,
        state_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward-backward under unary-only monotone CRF score:

            S(z) = sum_t log_b[t, z_t]

        Allowed stay/advance transitions have zero score.
        Disallowed transitions are excluded by state_valid/topology.

        Returns:
            gamma
            log_alpha
            log_beta
            log_gamma
            log_z
            log_b
        """
        B, T_speech, T_text = raw_unary.shape
        device = raw_unary.device

        log_b = self._compute_log_unary_potential(
            evidence=raw_unary,
            state_valid=state_valid,
        ).contiguous()

        # ------------------------------------------------------------------
        # Forward:
        # alpha[t, j] = log score of all partial paths ending at z_t=j.
        #
        # alpha[t,j] =
        #     log_b[t,j]
        #     +
        #     logaddexp(alpha[t-1,j], alpha[t-1,j-1])
        # ------------------------------------------------------------------
        log_alpha_steps: list[torch.Tensor] = []

        alpha_t = self.__initial_dp_score(
            log_b=log_b,
            state_valid=state_valid,
        )
        log_alpha_steps.append(alpha_t)

        for t in range(1, T_speech):  # forward-recursion
            stay, adv = self.__prev_to_current_scores(
                prev_score=alpha_t,
            )

            alpha_t = log_b[:, t, :] + torch.logaddexp(stay, adv)
            alpha_t = alpha_t.masked_fill(~state_valid[:, t, :], self.neg_large)

            log_alpha_steps.append(alpha_t)

        log_alpha = torch.stack(log_alpha_steps, dim=1)

        # ------------------------------------------------------------------
        # Backward:
        #
        # beta[t,j] =
        #     logaddexp(
        #         log_b[t+1,j]   + beta[t+1,j],
        #         log_b[t+1,j+1] + beta[t+1,j+1]
        #     )
        # ------------------------------------------------------------------
        log_beta_steps: list[torch.Tensor | None] = [None] * T_speech
        beta_next: torch.Tensor | None = None

        j = torch.arange(T_text, device=device).view(1, T_text)
        terminal_state = j == (text_lengths - 1).view(B, 1)

        for t in range(T_speech - 1, -1, -1):  # backward-recursion
            base = raw_unary.new_full((B, T_text), self.neg_large)

            terminal_batches = (spec_lengths - 1) == t
            base = torch.where(
                terminal_batches.view(B, 1) & terminal_state,
                torch.zeros_like(base),
                base,
            )

            if t == T_speech - 1:
                beta_t = base
            else:
                assert beta_next is not None

                stay = log_b[:, t + 1, :] + beta_next

                adv = torch.cat(
                    [
                        log_b[:, t + 1, 1:] + beta_next[:, 1:],
                        raw_unary.new_full((B, 1), self.neg_large),
                    ],
                    dim=1,
                )

                recursive = torch.logaddexp(stay, adv)

                beta_t = torch.where(
                    terminal_batches.view(B, 1),
                    base,
                    recursive,
                )

            beta_t = beta_t.masked_fill(~state_valid[:, t, :], self.neg_large)

            log_beta_steps[t] = beta_t
            beta_next = beta_t

        log_beta = torch.stack(log_beta_steps, dim=1)  # type: ignore[arg-type]

        batch_idx = torch.arange(B, device=device)
        log_z = log_alpha[batch_idx, spec_lengths - 1, text_lengths - 1]

        raw_log_gamma = log_alpha + log_beta - log_z.view(B, 1, 1)
        raw_log_gamma = raw_log_gamma.masked_fill(~state_valid, self.neg_large)

        # Row-normalize posterior marginals for numerical stability.
        valid_frame = state_valid.any(dim=-1, keepdim=True)
        row_log_norm = torch.logsumexp(raw_log_gamma, dim=-1, keepdim=True)

        log_gamma = torch.where(
            valid_frame,
            raw_log_gamma - row_log_norm,
            raw_log_gamma,
        )
        log_gamma = log_gamma.masked_fill(~state_valid, self.neg_large)

        gamma = torch.exp(log_gamma).masked_fill(~state_valid, 0.0)

        return (
            gamma,
            log_alpha,
            log_beta,
            log_gamma,
            log_z,
            log_b,
        )

    def _viterbi_decode(
        self,
        raw_unary: torch.Tensor,
        state_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAP path decoding under the same unary-only CRF score:

            z* = argmax_z sum_t log_b[t, z_t]

        subject to strict monotone topology.
        """
        B, T_speech, T_text = raw_unary.shape
        device = raw_unary.device

        log_b = self._compute_log_unary_potential(
            evidence=raw_unary,
            state_valid=state_valid,
        )

        delta_steps: list[torch.Tensor] = []
        backptr = torch.zeros(
            B,
            T_speech,
            T_text,
            dtype=torch.long,
            device=device,
        )

        delta_t = self.__initial_dp_score(
            log_b=log_b,
            state_valid=state_valid,
        )
        delta_steps.append(delta_t)

        j_idx = torch.arange(T_text, device=device).view(1, T_text).expand(B, -1)
        prev_stay = j_idx
        prev_adv = torch.clamp(j_idx - 1, min=0)

        for t in range(1, T_speech):  # forward-recursion with backpointer
            stay, adv = self.__prev_to_current_scores(
                prev_score=delta_t,
            )

            choose_adv = adv > stay
            best_prev_score = torch.where(choose_adv, adv, stay)

            delta_t = log_b[:, t, :] + best_prev_score
            delta_t = delta_t.masked_fill(~state_valid[:, t, :], self.neg_large)

            backptr[:, t, :] = torch.where(choose_adv, prev_adv, prev_stay)

            delta_steps.append(delta_t)

        log_delta = torch.stack(delta_steps, dim=1)

        batch_idx = torch.arange(B, device=device)
        viterbi_logp = log_delta[batch_idx, spec_lengths - 1, text_lengths - 1]

        path = torch.full(
            (B, T_speech),
            -1,
            dtype=torch.long,
            device=device,
        )
        hard_attn = raw_unary.new_zeros((B, T_speech, T_text))

        for b in range(B):  # backtracking
            t_end = int(spec_lengths[b].item()) - 1
            j_cur = int(text_lengths[b].item()) - 1

            for t in range(t_end, -1, -1):
                path[b, t] = j_cur
                hard_attn[b, t, j_cur] = 1.0

                if t > 0:
                    j_cur = int(backptr[b, t, j_cur].item())

        hard_attn = hard_attn.masked_fill(~state_valid, 0.0)
        hard_durations = hard_attn.sum(dim=1)

        return hard_attn, hard_durations, path, viterbi_logp

    def _compute_log_unary_potential(
        self,
        evidence: torch.Tensor,  # (B, T_s, T_t)
        state_valid: torch.Tensor,  # (B, T_s, T_t)
    ) -> torch.Tensor:
        """
        Compute normalized unary log potential log_b[t, j].

        Supports:
            local:
                denominator over a local text window around j

            global:
                denominator over all valid text states at frame t

        Returns:
            log_b: (B, T_s, T_t)
        """
        if self.unary_support_type == "local":
            return self.__local_support_log_potential(
                evidence=evidence,
                state_valid=state_valid,
            )

        if self.unary_support_type == "global":
            return self.__global_support_log_potential(
                evidence=evidence,
                state_valid=state_valid,
            )

        raise RuntimeError(f"Unexpected unary_support_type: {self.unary_support_type!r}")

    def __global_support_log_potential(
        self,
        evidence: torch.Tensor,  # (B, T_s, T_t)
        state_valid: torch.Tensor,  # (B, T_s, T_t)
    ) -> torch.Tensor:
        """
        Global-support unary log potential.

        For each frame t, the denominator is computed over all valid text states.

            log_phi[t, j]
            =
            e[t, j] / tau
            -
            logsumexp_{k: valid(t,k)} e[t, k] / tau

        Returns:
            log_phi: (B, T_s, T_t)
        """
        temperature = max(float(self.unary_temperature), 1e-6)

        score = evidence / temperature
        score_masked = score.masked_fill(~state_valid, self.neg_large)

        log_denom = torch.logsumexp(
            score_masked,
            dim=-1,
            keepdim=True,
        )  # (B, T_s, 1)

        log_phi = score_masked - log_denom
        log_phi = log_phi.masked_fill(~state_valid, self.neg_large)

        return log_phi

    def __local_support_log_potential(
        self,
        evidence: torch.Tensor,  # (B, T_s, T_t)
        state_valid: torch.Tensor,  # (B, T_s, T_t)
    ) -> torch.Tensor:
        """
        Local-support unary log potential.

        For each pair (t, j), the denominator is computed over a local text window
        around j. Invalid states are excluded from the denominator.

            log_phi[t, j]
            =
            e[t, j] / tau
            -
            logsumexp_{k in local_window(j), valid(t,k)} e[t, k] / tau

        Returns:
            log_phi: (B, T_s, T_t)
        """
        if self.unary_local_radius < 0:
            raise ValueError(
                f"Radius must be non-negative for local-support log potential, got {self.unary_local_radius}."
            )
        if self.unary_temperature <= 0:
            raise ValueError(
                f"Temperature must be positive for local-support log potential, got {self.unary_temperature}."
            )

        _, _, T_t = evidence.shape

        temperature = max(float(self.unary_temperature), 1e-6)

        score = evidence / temperature
        score_masked = score.masked_fill(~state_valid, self.neg_large)

        W = min(2 * int(self.unary_local_radius) + 1, T_t)

        j_idx = torch.arange(T_t, device=evidence.device)
        start_idx = torch.clamp(j_idx - int(self.unary_local_radius), min=0, max=T_t - W)

        # (B, T_s, T_t - W + 1, W)
        all_windows = score_masked.unfold(dimension=-1, size=W, step=1).contiguous()

        # (B, T_s, T_t, W)
        selected_windows = all_windows[:, :, start_idx, :]

        log_denom = torch.logsumexp(selected_windows, dim=-1)  # (B, T_s, T_t)

        log_phi = score_masked - log_denom
        log_phi = log_phi.masked_fill(~state_valid, self.neg_large)

        return log_phi

    def __initial_dp_score(
        self,
        log_b: torch.Tensor,
        state_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Common initialization for forward alpha and Viterbi delta.

        z_0 is forced to be token 0.
        """
        B, _, T_text = log_b.shape

        score = log_b.new_full((B, T_text), self.neg_large)
        score[:, 0] = log_b[:, 0, 0]
        score = score.masked_fill(~state_valid[:, 0, :], self.neg_large)

        return score

    def __prev_to_current_scores(
        self,
        prev_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Common stay/advance candidate scores from previous frame to current frame.

        Since transition scores are removed:

            stay    = prev_score[j]
            advance = prev_score[j-1]
        """
        B = prev_score.size(0)

        stay = prev_score

        adv = torch.cat(
            [
                prev_score.new_full((B, 1), self.neg_large),
                prev_score[:, :-1],
            ],
            dim=1,
        )

        return stay, adv

    @property
    def neg_large(self) -> float:
        if torch.get_default_dtype() in (torch.float16, torch.bfloat16):
            return -1e4
        return -1e9

    @staticmethod
    def _strict_reachability_mask(
        spec_lengths: torch.Tensor,  # (B,)
        text_lengths: torch.Tensor,  # (B,)
        T_speech: int,
        T_text: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Reachability mask for positive-duration strict monotone paths.

        A state j at frame t is valid if:
          1. j can be reached from state 0 by time t.
          2. final state N-1 can still be reached by final frame T-1.

        Zero-indexed:
            z_0 = 0
            z_{T-1} = N-1
            z_{t+1} in {z_t, z_t + 1}
        """
        t = torch.arange(T_speech, device=device).view(1, T_speech, 1)
        j = torch.arange(T_text, device=device).view(1, 1, T_text)

        T = spec_lengths.view(-1, 1, 1)
        N = text_lengths.view(-1, 1, 1)

        reachable_from_start = j <= t
        completable_to_end = (N - 1 - j) <= (T - 1 - t)

        within_lengths = (t < T) & (j < N)

        return reachable_from_start & completable_to_end & within_lengths
