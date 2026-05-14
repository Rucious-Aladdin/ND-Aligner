import math
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def local_support_log_potential(
    evidence: torch.Tensor,  # (B, T_s, T_t)
    state_valid: torch.Tensor,  # (B, T_s, T_t)
    *,
    radius: int = 10,
    temperature: float = 1.5,
    neg_large: float | None = None,
) -> torch.Tensor:

    _, _, T_t = evidence.shape

    if neg_large is None:
        neg_large = -1e4 if evidence.dtype in (torch.float16, torch.bfloat16) else -1e9

    score = evidence / float(temperature)

    # 1. 윈도우 크기 설정 (텍스트가 윈도우보다 짧으면 텍스트 길이로 맞춤)
    W = min(2 * radius + 1, T_t)

    # 2. [핵심] 네 아이디어의 수학적 구현: 각 토큰 j가 사용할 윈도우의 '시작 인덱스' 계산
    # 기본적으로 j - r 에서 시작하되,
    # 왼쪽 벽을 벗어나면 0으로 밀어버리고 (min=0)
    # 오른쪽 벽을 벗어나면 T_t - W 로 밀어버림 (max=T_t - W)
    j_idx = torch.arange(T_t, device=evidence.device)
    start_idx = torch.clamp(j_idx - radius, min=0, max=T_t - W)

    # 3. Unfold를 이용해 가능한 모든 크기 W의 연속된 윈도우를 뽑음
    # 결과 모양: (B, T_s, T_t - W + 1, W)
    all_windows = score.unfold(dimension=-1, size=W, step=1)

    # 4. 방금 계산한 start_idx를 이용해 각 j마다 자기가 들어가야 할 윈도우를 쏙쏙 빼옴
    # advanced indexing: (B, T_s, T_t, W)
    selected_windows = all_windows[:, :, start_idx, :]

    # 5. 이제 모든 토큰이 정확히 W개의 실제 경쟁자들을 가졌으니 LogSumExp!
    log_denom = torch.logsumexp(selected_windows, dim=-1)  # (B, T_s, T_t)

    log_phi = score - log_denom

    # 마지막으로 도달 불가능한 경로 마스킹
    log_phi = log_phi.masked_fill(~state_valid, neg_large)

    return log_phi


class ConvNormAct2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: tuple[int, int] = (3, 3),
        padding: tuple[int, int] = (1, 1),
        groups: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.norm = nn.GroupNorm(
            num_groups=_valid_group_count(dim_out, groups),
            num_channels=dim_out,
        )
        self.act = nn.GELU()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        groups: int = 8,
        kernel_size: tuple[int, int] = (3, 1),
        padding: tuple[int, int] = (1, 0),
    ):
        super().__init__()

        self.block1 = ConvNormAct2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )
        self.block2 = ConvNormAct2d(
            dim_out,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
        )

        if dim_in != dim_out:
            self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block2(self.block1(x)) + self.res_conv(x)


class Downsample(nn.Module):
    """
    Downsample only along speech/time axis.

    Input : (B, C, T_s, T_t)
    Output: (B, C_out, ceil(T_s / 2), T_t)

    Text axis T_t is not downsampled.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsample only along speech/time axis.
    Text axis T_t is not downsampled.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

    @override
    def forward(
        self,
        x: torch.Tensor,
        target_speech_len: int,
    ) -> torch.Tensor:
        x = F.interpolate(
            x,
            size=(target_speech_len, x.size(-1)),
            mode="nearest",
        )
        return self.conv(x)


class UNet2d(nn.Module):
    """
    U-Net over pairwise speech-text feature.

    - input : (B, H, T_s, T_t)
    - output: (B, 1, T_s, T_t)

    Downsampling is applied only along T_s:
        T_s -> T_s / 2 -> T_s / 4

    T_t is never downsampled.
    """

    def __init__(
        self,
        dim_in: int,
        base_dim: int = 16,
        groups: int = 8,
    ):
        super().__init__()

        d0 = base_dim
        d1 = base_dim * 2
        d2 = base_dim * 2

        # Text RF contribution: +4
        self.in_block = ResBlock2d(
            dim_in,
            d0,
            groups=groups,
            kernel_size=(3, 3),
            padding=(1, 1),
        )

        self.down1 = nn.Sequential(
            Downsample(d0, d1),
            ResBlock2d(
                d1,
                d1,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
        )

        self.down2 = nn.Sequential(
            Downsample(d1, d2),
            ResBlock2d(
                d2,
                d2,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
        )

        self.mid = nn.Sequential(
            ResBlock2d(
                d2,
                d2,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
            ResBlock2d(
                d2,
                d2,
                groups=groups,
                kernel_size=(3, 1),
                padding=(1, 0),
            ),
        )

        self.up1 = Upsample(d2, d1)
        self.dec1 = ResBlock2d(
            d1 + d1,
            d1,
            groups=groups,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

        self.up2 = Upsample(d1, d0)
        self.dec2 = ResBlock2d(
            d0 + d0,
            d0,
            groups=groups,
            kernel_size=(3, 1),
            padding=(1, 0),
        )

        # Text RF contribution: +2
        # Total text RF = 1 + 4 + 2 = 7 tokens.
        self.out_block = nn.Sequential(
            ConvNormAct2d(
                d0,
                d0,
                kernel_size=(3, 3),
                padding=(1, 1),
                groups=groups,
            ),
            nn.Conv2d(d0, 1, kernel_size=1),
        )

    @staticmethod
    def _pad_speech_to_multiple(
        x: torch.Tensor,
        multiple: int = 4,
    ) -> tuple[torch.Tensor, int]:
        T_s = x.size(-2)
        pad_s = (multiple - (T_s % multiple)) % multiple

        if pad_s > 0:
            # F.pad order for 4D is (left, right, top, bottom).
            # Here top/bottom correspond to the T_s axis.
            x = F.pad(x, (0, 0, 0, pad_s))

        return x, pad_s

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: pairwise feature, (B, H, T_s, T_t)

        Returns:
            score: (B, 1, T_s, T_t)
        """
        orig_T_s = x.size(-2)

        x, pad_s = self._pad_speech_to_multiple(x, multiple=4)

        h0 = self.in_block(x)
        h1 = self.down1(h0)
        h2 = self.down2(h1)

        x = self.mid(h2)

        x = self.up1(x, target_speech_len=h1.size(-2))
        x = torch.cat([x, h1], dim=1)
        x = self.dec1(x)

        x = self.up2(x, target_speech_len=h0.size(-2))
        x = torch.cat([x, h0], dim=1)
        x = self.dec2(x)

        x = self.out_block(x)

        if pad_s > 0:
            x = x[:, :, :orig_T_s, :]

        return x


class UNetEmissionPredictor(nn.Module):
    """
    U-Net emission / unary evidence predictor.

    Pairwise input:
        [
            f_s(h_spec_t) ;
            f_x(h_text_j) ;
            f_c(cond)
        ]

    No positional feature is used here. Global ordering is handled by the
    monotonic latent-path constraint, not by the emission network.

    Returns:
        evidence: raw pairwise score e[t, j], (B, T_s, T_t)
    """

    def __init__(
        self,
        dim_spec: int,
        dim_text: int,
        dim_latent: int,
        dim_cond: int,
        cond_channels: int = 32,
        base_dim: int = 16,
        groups: int = 8,
        emission_scale_init: float = -2.0,
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
            dim_in=2 * dim_latent + cond_channels,
            base_dim=base_dim,
            groups=groups,
        )

        # Keeps early evidence small, so the transition prior is not overwritten
        # too aggressively at the beginning of training.
        self.evidence_scale = nn.Parameter(
            torch.tensor(float(emission_scale_init), dtype=torch.float32)
        )

    @override
    def forward(
        self,
        h_spec: torch.Tensor,  # (B, C_s, T_s)
        h_text: torch.Tensor,  # (B, C_t, T_t)
        cond: torch.Tensor,  # (B, D_cond)
    ) -> torch.Tensor:
        """
        Returns:
            evidence: (B, T_s, T_t)
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

        pairwise = torch.cat(
            [spec_pair, text_pair, cond_pair],
            dim=-1,
        )  # (B, T_s, T_t, 2C + cond_channels)

        pairwise = pairwise.permute(0, 3, 1, 2).contiguous()

        emission = self.unet(pairwise).squeeze(1)
        emission = emission * self.evidence_scale.exp()

        return emission


class MonotonicCRFAligner(nn.Module):
    def __init__(
        self,
        dim_spec: int,
        dim_text: int,
        dim_hidden: int,
        dim_cond: int,
        dim_emit_latent: int,
        transition_adv_init: float = 0.15,
        emission_radius: int = 10,
        emission_temperature: float = 1.0,
        cond_channels: int = 32,
        base_dim: int = 16,
        groups: int = 8,
        evidence_scale_init: float = -2.0,
    ):
        super().__init__()

        self.dim_spec = dim_spec
        self.dim_text = dim_text
        self.dim_hidden = dim_hidden
        self.dim_emit_latent = dim_emit_latent

        self.emission_local_radius = int(emission_radius)
        self.emission_temperature = float(emission_temperature)

        self.emission_predictor = UNetEmissionPredictor(
            dim_spec=dim_spec,
            dim_text=dim_text,
            dim_latent=dim_emit_latent,
            dim_cond=dim_cond,
            cond_channels=cond_channels,
            base_dim=base_dim,
            groups=groups,
            emission_scale_init=evidence_scale_init,
        )

        transition_adv_init = min(max(float(transition_adv_init), 1e-5), 1.0 - 1e-5)
        kappa_adv_init = math.log(transition_adv_init) - math.log1p(-transition_adv_init)

        # The only transition parameter.
        self.kappa_adv = nn.Parameter(torch.tensor(kappa_adv_init, dtype=torch.float32))

    @staticmethod
    def _neg_large(dtype: torch.dtype) -> float:
        if dtype in (torch.float16, torch.bfloat16):
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
        """
        t = torch.arange(T_speech, device=device).view(1, T_speech, 1)
        j = torch.arange(T_text, device=device).view(1, 1, T_text)

        T = spec_lengths.view(-1, 1, 1)
        N = text_lengths.view(-1, 1, 1)

        reachable_from_start = j <= t
        completable_to_end = (N - 1 - j) <= (T - 1 - t)

        within_lengths = (t < T) & (j < N)

        return reachable_from_start & completable_to_end & within_lengths

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

        # 1. Raw pairwise emission evidence e[t, j].
        raw_emission = self.emission_predictor(
            h_spec=h_spec,
            h_text=h_text,
            cond=cond,
        )  # (B, T_s, T_t)

        _, T_speech, T_text = raw_emission.shape
        device = raw_emission.device

        length_valid = spec_mask.unsqueeze(2) & text_mask.unsqueeze(1)
        reachable = self._strict_reachability_mask(
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
            T_speech=T_speech,
            T_text=T_text,
            device=device,
        )

        state_valid = length_valid & reachable

        neg_large = self._neg_large(raw_emission.dtype)

        masked_raw_emission = raw_emission.masked_fill(~state_valid, neg_large)

        # 2. Forward-backward.
        (
            gamma,
            log_alpha,
            log_beta,
            log_gamma,
            raw_log_z,
            log_b,
            log_a_stay,
            log_a_adv,
        ) = self._forward_backward(
            raw_emission=raw_emission,
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
            raw_emission,
            masked_raw_emission,
            log_b,
            log_a_stay,
            log_a_adv,
            durations,
        )

        if not return_hard:
            return base_outputs

        # 3. Optional Viterbi hard path.
        hard_gamma, hard_durations, viterbi_path, viterbi_logp = self._viterbi_decode(
            raw_emission=raw_emission,
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

    def _transition_log_probs(
        self,
        batch_size: int,
        T_speech: int,
        T_text: int,
        text_lengths: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_a_stay[b, t, j] = log score(z_{t+1}=j   | z_t=j)
            log_a_adv [b, t, j] = log score(z_{t+1}=j+1 | z_t=j)

        Transition is controlled only by scalar kappa_adv.
        """
        neg_large = self._neg_large(dtype)

        adv_logit = self.kappa_adv.to(dtype=dtype, device=device)

        log_adv_scalar = F.logsigmoid(adv_logit)
        log_stay_scalar = F.logsigmoid(-adv_logit)

        log_a_adv = log_adv_scalar.expand(batch_size, T_speech, T_text).clone()
        log_a_stay = log_stay_scalar.expand(batch_size, T_speech, T_text).clone()

        j = torch.arange(T_text, device=device).view(1, 1, T_text)
        within = j < text_lengths.view(batch_size, 1, 1)

        log_a_stay = log_a_stay.masked_fill(~within, neg_large)
        log_a_adv = log_a_adv.masked_fill(~within, neg_large)

        return log_a_stay, log_a_adv

    def _forward_backward(
        self,
        raw_emission: torch.Tensor,
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
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward-backward under the monotonic latent-path CRF score:

            sum_t log_b[t, z_t]
            +
            sum_t log_a[z_t -> z_{t+1}]

        Here, log_b is a locally normalized unary potential, not a
        generative emission probability.
        """
        B, T_speech, T_text = raw_emission.shape
        device = raw_emission.device

        log_b, log_a_stay, log_a_adv, neg_large = self.__prepare_dp_scores(
            raw_emission=raw_emission,
            state_valid=state_valid,
            text_lengths=text_lengths,
        )

        # ------------------------------------------------------------------
        # Forward:
        # alpha[t, j] = log score of all partial paths ending at z_t=j
        # ------------------------------------------------------------------
        log_alpha_steps: list[torch.Tensor] = []

        alpha_t = self.__initial_dp_score(
            log_b=log_b,
            state_valid=state_valid,
            neg_large=neg_large,
        )
        log_alpha_steps.append(alpha_t)

        for t in range(1, T_speech):
            stay, adv = self.__prev_to_current_scores(
                prev_score=alpha_t,
                log_a_stay_t=log_a_stay[:, t - 1, :],
                log_a_adv_t=log_a_adv[:, t - 1, :],
                neg_large=neg_large,
            )

            alpha_t = log_b[:, t, :] + torch.logaddexp(stay, adv)
            alpha_t = alpha_t.masked_fill(~state_valid[:, t, :], neg_large)
            log_alpha_steps.append(alpha_t)

        log_alpha = torch.stack(log_alpha_steps, dim=1)

        # ------------------------------------------------------------------
        # Backward:
        # beta[t, j] = log score of all suffix paths from z_t=j
        # ------------------------------------------------------------------
        log_beta_steps: list[torch.Tensor | None] = [None] * T_speech
        beta_next: torch.Tensor | None = None

        j = torch.arange(T_text, device=device).view(1, T_text)
        terminal_state = j == (text_lengths - 1).view(B, 1)

        for t in range(T_speech - 1, -1, -1):
            base = raw_emission.new_full((B, T_text), neg_large)

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

                stay = log_a_stay[:, t, :] + log_b[:, t + 1, :] + beta_next

                adv = torch.cat(
                    [
                        log_a_adv[:, t, :-1] + log_b[:, t + 1, 1:] + beta_next[:, 1:],
                        raw_emission.new_full((B, 1), neg_large),
                    ],
                    dim=1,
                )

                recursive = torch.logaddexp(stay, adv)

                beta_t = torch.where(
                    terminal_batches.view(B, 1),
                    base,
                    recursive,
                )

            beta_t = beta_t.masked_fill(~state_valid[:, t, :], neg_large)
            log_beta_steps[t] = beta_t
            beta_next = beta_t

        log_beta = torch.stack(log_beta_steps, dim=1)  # type: ignore[arg-type]

        batch_idx = torch.arange(B, device=device)
        log_z = log_alpha[batch_idx, spec_lengths - 1, text_lengths - 1]

        raw_log_gamma = log_alpha + log_beta - log_z.view(B, 1, 1)
        raw_log_gamma = raw_log_gamma.masked_fill(~state_valid, neg_large)

        log_gamma = raw_log_gamma
        gamma = torch.exp(log_gamma).masked_fill(~state_valid, 0.0)

        return (
            gamma,
            log_alpha,
            log_beta,
            log_gamma,
            log_z,
            log_b,
            log_a_stay,
            log_a_adv,
        )

    def _viterbi_decode(
        self,
        raw_emission: torch.Tensor,
        state_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAP path decoding under the same CRF score used by forward-sum.
        """
        B, T_speech, T_text = raw_emission.shape
        device = raw_emission.device

        log_b, log_a_stay, log_a_adv, neg_large = self.__prepare_dp_scores(
            raw_emission=raw_emission,
            state_valid=state_valid,
            text_lengths=text_lengths,
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
            neg_large=neg_large,
        )
        delta_steps.append(delta_t)

        j_idx = torch.arange(T_text, device=device).view(1, T_text).expand(B, -1)
        prev_stay = j_idx
        prev_adv = torch.clamp(j_idx - 1, min=0)

        for t in range(1, T_speech):
            stay, adv = self.__prev_to_current_scores(
                prev_score=delta_t,
                log_a_stay_t=log_a_stay[:, t - 1, :],
                log_a_adv_t=log_a_adv[:, t - 1, :],
                neg_large=neg_large,
            )

            choose_adv = adv > stay
            best_prev_score = torch.where(choose_adv, adv, stay)

            delta_t = log_b[:, t, :] + best_prev_score
            delta_t = delta_t.masked_fill(~state_valid[:, t, :], neg_large)

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
        hard_attn = raw_emission.new_zeros((B, T_speech, T_text))

        for b in range(B):
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

    def __prepare_dp_scores(
        self,
        raw_emission: torch.Tensor,
        state_valid: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Common preparation for forward-backward and Viterbi.

        Returns:
            log_b      : (B, T_speech, T_text)
            log_a_stay: (B, T_speech, T_text)
            log_a_adv : (B, T_speech, T_text)
            neg_large : large negative sentinel

        log_b is a locally normalized unary potential:

            log_b[t, j]
            =
            e[t, j] / tau
            -
            logsumexp_{k in [j-r, j+r]} e[t, k] / tau

        It is not required to sum to one globally over text states.
        """
        dtype = raw_emission.dtype
        device = raw_emission.device
        neg_large = self._neg_large(dtype)

        B, T_speech, T_text = raw_emission.shape

        log_b = local_support_log_potential(
            evidence=raw_emission,
            state_valid=state_valid,
            radius=self.emission_local_radius,
            temperature=self.emission_temperature,
            neg_large=neg_large,
        )
        # score = raw_emission / self.emission_temperature
        # log_b = score.masked_fill(~state_valid, neg_large)

        log_a_stay, log_a_adv = self._transition_log_probs(
            batch_size=B,
            T_speech=T_speech,
            T_text=T_text,
            text_lengths=text_lengths,
            dtype=dtype,
            device=device,
        )

        return log_b, log_a_stay, log_a_adv, neg_large

    def __initial_dp_score(
        self,
        log_b: torch.Tensor,
        state_valid: torch.Tensor,
        neg_large: float,
    ) -> torch.Tensor:
        """
        Common initialization for forward alpha and Viterbi delta.

        z_0 is forced to be token 0.
        """
        B, _, T_text = log_b.shape

        score = log_b.new_full((B, T_text), neg_large)
        score[:, 0] = log_b[:, 0, 0]
        score = score.masked_fill(~state_valid[:, 0, :], neg_large)

        return score

    def __prev_to_current_scores(
        self,
        prev_score: torch.Tensor,
        log_a_stay_t: torch.Tensor,
        log_a_adv_t: torch.Tensor,
        neg_large: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Common stay/advance candidate scores from previous frame to current frame.

        stay:
            z_{t-1}=j -> z_t=j

        advance:
            z_{t-1}=j-1 -> z_t=j
        """
        B = prev_score.size(0)

        stay = prev_score + log_a_stay_t

        adv = torch.cat(
            [
                prev_score.new_full((B, 1), neg_large),
                prev_score[:, :-1] + log_a_adv_t[:, :-1],
            ],
            dim=1,
        )

        return stay, adv
