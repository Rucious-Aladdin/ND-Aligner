import ctypes
import logging
from pathlib import Path
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


def valid_group_count(channels: int, max_groups: int = 8) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResidualConvNormAct2d(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: tuple[int, int] = (3, 3),
        groups: int = 8,
    ):
        super().__init__()

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("kernel_size must contain odd values.")

        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            dim_in,
            dim_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.norm = nn.GroupNorm(
            num_groups=valid_group_count(dim_out, groups),
            num_channels=dim_out,
        )

        self.act = nn.GELU()

        if dim_in == dim_out:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=1,
            )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)

        return h + self.skip(x)


class Conv2dNet(nn.Module):
    """
    Plain residual Conv2D network over pairwise speech-text feature.

    Input : (B, C_in, T_s, T_t)
    Output: (B, C_out, T_s, T_t)

    This preserves both T_s and T_t lengths.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int = 16,
        dim_out: int = 1,
        num_layers: int = 3,
        kernel_size: tuple[int, int] = (3, 3),
        groups: int = 8,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []

        layers.append(
            ResidualConvNormAct2d(
                dim_in=dim_in,
                dim_out=dim_hidden,
                kernel_size=kernel_size,
                groups=groups,
            )
        )

        for _ in range(num_layers - 1):
            layers.append(
                ResidualConvNormAct2d(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    kernel_size=kernel_size,
                    groups=groups,
                )
            )

        # Do NOT residual-connect this final projection.
        layers.append(
            nn.Conv2d(
                dim_hidden,
                dim_out,
                kernel_size=1,
            )
        )

        self.net = nn.Sequential(*layers)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NodePotentialPredictor(nn.Module):
    """
    unary potential predictor.

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
        dim_hidden: int = 16,
        groups: int = 8,
        unary_scale_init: float = -2.0,
        unary_network_type: str = "conv",
        conv_num_layers: int = 4,
        conv_kernel_size: tuple[int, int] = (3, 3),
    ):
        super().__init__()

        if unary_network_type not in ("conv", "negative-l2"):
            raise ValueError(
                "unary_network_type must be one of "
                + f"'conv', 'negative-l2', got {unary_network_type!r}."
            )

        self.unary_network_type = unary_network_type
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

        if unary_network_type == "conv":
            if self.cond_channels > 0:
                self.cond_proj = nn.Linear(dim_cond, self.cond_channels)
            else:
                self.cond_proj = None

            in_dim = 2 * dim_latent + self.cond_channels

            self.score_net = Conv2dNet(
                dim_in=in_dim,
                dim_hidden=dim_hidden,
                dim_out=1,
                num_layers=conv_num_layers,
                kernel_size=conv_kernel_size,
                groups=groups,
            )
        else:
            self.cond_proj = None
            self.score_net = None

        # Keeps early evidence small.
        self.evidence_scale = nn.Parameter(
            torch.tensor(float(unary_scale_init), dtype=torch.float32)
        )

    @override
    def forward(
        self,
        h_spec: torch.Tensor,  # (B, C_s, T_s)
        h_text: torch.Tensor,  # (B, C_t, T_t)
        cond: torch.Tensor,  # (B, D_cond)
    ) -> torch.Tensor:
        h_spec_t = h_spec.transpose(1, 2).contiguous()  # (B, T_s, C_s)
        h_text_t = h_text.transpose(1, 2).contiguous()  # (B, T_t, C_t)

        h_spec_t = self.spec_proj(h_spec_t)
        h_text_t = self.text_proj(h_text_t)

        B, T_s, C = h_spec_t.shape
        _, T_t, _ = h_text_t.shape

        if self.unary_network_type == "negative-l2":
            diff = h_spec_t.unsqueeze(2) - h_text_t.unsqueeze(1)
            unary_potential = -(diff.square().sum(dim=-1))
            unary_potential = unary_potential * self.evidence_scale.exp()
            return unary_potential

        assert self.score_net is not None

        spec_pair = h_spec_t.unsqueeze(2).expand(B, T_s, T_t, C)
        text_pair = h_text_t.unsqueeze(1).expand(B, T_s, T_t, C)

        pairwise_parts: list[torch.Tensor] = [
            spec_pair,
            text_pair,
        ]

        if self.cond_proj is not None:
            cond_feat = self.cond_proj(cond)
            cond_pair = cond_feat.view(B, 1, 1, self.cond_channels).expand(
                B,
                T_s,
                T_t,
                self.cond_channels,
            )
            pairwise_parts.append(cond_pair)

        pairwise = torch.cat(pairwise_parts, dim=-1)
        pairwise = pairwise.permute(0, 3, 1, 2).contiguous()
        # (B, C_pair, T_s, T_t)

        unary_potential = self.score_net(pairwise).squeeze(1)
        unary_potential = unary_potential * self.evidence_scale.exp()

        return unary_potential


_VITERBI_LIBRARY: ctypes.CDLL | None = None
_VITERBI_LIBRARY_ERROR: Exception | None = None
_VITERBI_FALLBACK_LOGGED = False
logger = logging.getLogger(__name__)


def _load_viterbi_library() -> ctypes.CDLL:
    """Load and configure the local C Viterbi shared library once."""
    global _VITERBI_LIBRARY
    global _VITERBI_LIBRARY_ERROR

    if _VITERBI_LIBRARY is not None:
        return _VITERBI_LIBRARY

    if _VITERBI_LIBRARY_ERROR is not None:
        raise RuntimeError(
            "The C Viterbi library previously failed to load."
        ) from _VITERBI_LIBRARY_ERROR

    library_path = Path(__file__).resolve().parent / "mas" / "viterbi_dp.so"

    try:
        library = ctypes.CDLL(str(library_path))
        function = library.viterbi_forward_backtrack_f32

        function.argtypes = [
            ctypes.c_void_p,  # log_b: float32 [B, T_speech, T_text]
            ctypes.c_void_p,  # dp_valid: uint8 [B, T_speech, T_text]
            ctypes.c_void_p,  # initial_delta: float32 [B, T_text]
            ctypes.c_void_p,  # spec_lengths: int64 [B]
            ctypes.c_void_p,  # text_lengths: int64 [B]
            ctypes.c_void_p,  # opt_sep_mask: uint8 [B, T_text] or NULL
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_void_p,  # path: int64 [B, T_speech]
            ctypes.c_void_p,  # viterbi_logp: float32 [B]
        ]
        function.restype = ctypes.c_int
    except Exception as exc:
        _VITERBI_LIBRARY_ERROR = exc  # type: ignore
        raise RuntimeError(f"Failed to load C Viterbi library: {library_path}") from exc

    _VITERBI_LIBRARY = library  # type: ignore
    return library


class LinearCRFAligner(nn.Module):
    """
    Unary-only monotonic latent-path CRF aligner.

    Path topology:
        z_0 = 0
        z_{T-1} = N-1
        z_t -> z_t      stay
        z_t -> z_t + 1  advance
        z_t -> z_t + 2  optional-separator skip
                         allowed iff token z_t + 1 is optional

    There is no learned transition score.
    Allowed transitions have score 0.
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
        # normalization and support configs
        unary_network_type: str = "conv",  # "conv", "negative-l2"
        unary_support_type: str = "global",  # "global" "raw", "bernoulli"
        unary_temperature: float = 1.0,
        unary_scale_init: float = -2.0,
        # unary network configs
        conv_num_layers: int = 4,
        conv_num_groups: int = 8,
        conv_dim_hidden: int = 16,
        conv_kernel_size: tuple[int, int] = (3, 3),
    ):
        super().__init__()

        self.dim_spec = dim_spec
        self.dim_text = dim_text

        self.unary_support_type = unary_support_type
        self.unary_temperature = float(unary_temperature)

        self.unary_predictor = NodePotentialPredictor(
            dim_spec=dim_spec,
            dim_text=dim_text,
            dim_latent=dim_unary_latent,
            dim_cond=dim_cond,
            cond_channels=cond_channels,
            dim_hidden=conv_dim_hidden,
            groups=conv_num_groups,
            unary_scale_init=unary_scale_init,
            unary_network_type=unary_network_type,
            conv_num_layers=conv_num_layers,
            conv_kernel_size=conv_kernel_size,
        )

        try:
            self.viterbi_lib = _load_viterbi_library()
        except:
            print("[WARNING] load-viterbi lib failed.")
            self.viterbi_lib = None

    @override
    def forward(
        self,
        h_spec: torch.Tensor,  # (B, C_s, T_s)
        spec_mask: torch.Tensor,  # (B, T_s) or (B, 1, T_s)
        h_text: torch.Tensor,  # (B, C_t, T_t)
        text_mask: torch.Tensor,  # (B, T_t) or (B, 1, T_t)
        cond: torch.Tensor,  # (B, D_cond)
        opt_sep_mask: torch.Tensor | None = None,  # (B, T_t) or (B, 1, T_t)
        return_soft: bool = True,
        return_hard: bool = False,
    ) -> tuple[torch.Tensor | None, ...]:
        if (not return_soft) and (not return_hard):
            raise RuntimeError("At least one of return_soft and return_hard must be True.")

        if spec_mask.dim() == 3:
            spec_mask = spec_mask.squeeze(1)
        if text_mask.dim() == 3:
            text_mask = text_mask.squeeze(1)

        spec_mask = spec_mask.bool()
        text_mask = text_mask.bool()

        spec_lengths = spec_mask.sum(dim=-1).long()
        text_lengths = text_mask.sum(dim=-1).long()

        if opt_sep_mask is not None:
            if opt_sep_mask.dim() == 3:
                opt_sep_mask = opt_sep_mask.squeeze(1)

            opt_sep_mask = opt_sep_mask.bool() & text_mask
        else:
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
        )  # (B, T_s, T_t)

        _, T_speech, T_text = raw_unary.shape
        device = raw_unary.device

        length_valid = spec_mask.unsqueeze(2) & text_mask.unsqueeze(1)

        if opt_sep_mask is None:
            reachable = self._strict_reachability_mask(
                spec_lengths=spec_lengths,
                text_lengths=text_lengths,
                T_speech=T_speech,
                T_text=T_text,
                device=device,
            )
        else:
            reachable = self._optional_separator_reachability_mask(
                spec_lengths=spec_lengths,
                text_lengths=text_lengths,
                opt_sep_mask=opt_sep_mask,
                T_speech=T_speech,
                T_text=T_text,
                device=device,
            )

        batch_idx = torch.arange(
            spec_lengths.size(0),
            device=device,
        )

        has_valid_path = reachable[
            batch_idx,
            spec_lengths - 1,
            text_lengths - 1,
        ]

        if not torch.all(has_valid_path) and opt_sep_mask is not None:
            invalid_batches = torch.nonzero(
                ~has_valid_path,
                as_tuple=False,
            ).flatten()

            details = [
                {
                    "batch": int(b),
                    "spec_length": int(spec_lengths[b]),
                    "text_length": int(text_lengths[b]),
                    "optional_count": int(opt_sep_mask[b, : text_lengths[b]].sum()),
                }
                for b in invalid_batches.detach().cpu().tolist()
            ]

            raise ValueError(
                "No valid monotone path exists under the optional-separator "
                + f"topology. Invalid samples: {details}"
            )

        dp_valid = length_valid & reachable

        masked_raw_unary = raw_unary.masked_fill(
            ~dp_valid,
            self.neg_large,
        )

        # ------------------------------------------------------------
        # Compute log_b. Both soft forward-backward and Viterbi use it.
        # ------------------------------------------------------------
        log_b = self._compute_log_node_potential(
            evidence=raw_unary,
            unary_valid=length_valid,
        ).contiguous()

        # ------------------------------------------------------------
        # Optional soft path: forward-backward posterior and CRF log Z.
        # ------------------------------------------------------------
        if return_soft:
            (
                gamma,
                log_alpha,
                log_beta,
                log_gamma,
                raw_log_z,
            ) = self._forward_backward(
                log_b=log_b,
                dp_valid=dp_valid,
                spec_lengths=spec_lengths,
                text_lengths=text_lengths,
                opt_sep_mask=opt_sep_mask,
            )

            gamma = gamma.masked_fill(~dp_valid, 0.0)
            durations = gamma.sum(dim=1)

            if self.unary_support_type == "bernoulli":
                bernoulli_penalty = self.__bernoulli_negative_penalty(
                    evidence=raw_unary,
                    penalty_valid=dp_valid,
                )
                raw_log_z = raw_log_z - bernoulli_penalty

            norm_log_z = raw_log_z / spec_lengths.float()
        else:
            # Hard-only mode. Soft posterior and CRF log-normalizer are intentionally
            # skipped to avoid the forward-backward DP cost.
            gamma = None
            log_alpha = None
            log_beta = None
            log_gamma = None
            raw_log_z = None
            norm_log_z = None
            durations = None

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
            log_b=log_b,
            dp_valid=dp_valid,
            spec_lengths=spec_lengths,
            text_lengths=text_lengths,
            opt_sep_mask=opt_sep_mask,
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
        log_b: torch.Tensor,
        dp_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        opt_sep_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward-backward using precomputed log_b.

        This is used so that hard-only decoding can compute log_b once and skip
        forward-backward entirely.
        """
        B, T_speech, T_text = log_b.shape
        device = log_b.device

        # ------------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------------
        log_alpha_steps: list[torch.Tensor] = []

        alpha_t = self.__initial_dp_score(
            log_b=log_b,
            dp_valid=dp_valid,
            opt_sep_mask=opt_sep_mask,
        )
        log_alpha_steps.append(alpha_t)

        for t in range(1, T_speech):
            stay, adv, skip = self.__prev_to_current_scores(
                prev_score=alpha_t,
                opt_sep_mask=opt_sep_mask,
            )
            if skip is None:
                prev_sum = torch.logaddexp(stay, adv)
            else:
                prev_sum = torch.logaddexp(torch.logaddexp(stay, adv), skip)

            alpha_t = log_b[:, t, :] + prev_sum
            alpha_t = alpha_t.masked_fill(~dp_valid[:, t, :], self.neg_large)

            log_alpha_steps.append(alpha_t)

        log_alpha = torch.stack(log_alpha_steps, dim=1)

        # ------------------------------------------------------------------
        # Backward
        # ------------------------------------------------------------------
        log_beta_steps: list[torch.Tensor | None] = [None] * T_speech
        beta_next: torch.Tensor | None = None

        j = torch.arange(T_text, device=device).view(1, T_text)
        terminal_state = j == (text_lengths - 1).view(B, 1)

        for t in range(T_speech - 1, -1, -1):
            base = log_b.new_full((B, T_text), self.neg_large)

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

                next_score = log_b[:, t + 1, :] + beta_next

                stay, adv, skip = self.__current_to_next_scores(
                    next_score=next_score,
                    opt_sep_mask=opt_sep_mask,
                )

                if skip is None:
                    recursive = torch.logaddexp(stay, adv)
                else:
                    recursive = torch.logaddexp(torch.logaddexp(stay, adv), skip)

                beta_t = torch.where(
                    terminal_batches.view(B, 1),
                    base,
                    recursive,
                )

            beta_t = beta_t.masked_fill(~dp_valid[:, t, :], self.neg_large)

            log_beta_steps[t] = beta_t
            beta_next = beta_t

        log_beta = torch.stack(log_beta_steps, dim=1)  # type: ignore[arg-type]

        batch_idx = torch.arange(B, device=device)
        log_z = log_alpha[batch_idx, spec_lengths - 1, text_lengths - 1]

        raw_log_gamma = log_alpha + log_beta - log_z.view(B, 1, 1)
        raw_log_gamma = raw_log_gamma.masked_fill(~dp_valid, self.neg_large)

        valid_frame = dp_valid.any(dim=-1, keepdim=True)
        row_log_norm = torch.logsumexp(raw_log_gamma, dim=-1, keepdim=True)

        log_gamma = torch.where(
            valid_frame,
            raw_log_gamma - row_log_norm,
            raw_log_gamma,
        )
        log_gamma = log_gamma.masked_fill(~dp_valid, self.neg_large)

        gamma = torch.exp(log_gamma).masked_fill(~dp_valid, 0.0)

        return (
            gamma,
            log_alpha,
            log_beta,
            log_gamma,
            log_z,
        )

    def _viterbi_decode(
        self,
        log_b: torch.Tensor,
        dp_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        opt_sep_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode with the C implementation and fall back to PyTorch on failure.

        The shared object is expected at:

            tts/models/modules/mas/viterbi_dp.so
        """
        global _VITERBI_FALLBACK_LOGGED

        try:
            return self.__viterbi_decode_c(
                log_b=log_b,
                dp_valid=dp_valid,
                spec_lengths=spec_lengths,
                text_lengths=text_lengths,
                opt_sep_mask=opt_sep_mask,
            )
        except Exception:
            if not _VITERBI_FALLBACK_LOGGED:
                logger.exception(
                    "C Viterbi decoding failed; falling back to the PyTorch implementation."
                )
                _VITERBI_FALLBACK_LOGGED = True  # type: ignore

            return self.__viterbi_decode_python(
                log_b=log_b,
                dp_valid=dp_valid,
                spec_lengths=spec_lengths,
                text_lengths=text_lengths,
                opt_sep_mask=opt_sep_mask,
            )

    def __viterbi_decode_c(
        self,
        log_b: torch.Tensor,
        dp_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        opt_sep_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward recursion and backtracking through ``viterbi_dp.so``."""
        B, T_speech, T_text = log_b.shape
        device = log_b.device

        library = _load_viterbi_library()
        function = library.viterbi_forward_backtrack_f32

        # The C ABI is fixed to float32, uint8, and int64.
        log_b_cpu = log_b.detach().to(device="cpu", dtype=torch.float32).contiguous()
        dp_valid_cpu = dp_valid.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        spec_lengths_cpu = spec_lengths.detach().to(device="cpu", dtype=torch.int64).contiguous()
        text_lengths_cpu = text_lengths.detach().to(device="cpu", dtype=torch.int64).contiguous()

        if opt_sep_mask is None:
            opt_sep_mask_cpu = None
            optional_separator_pointer = ctypes.c_void_p()
        else:
            opt_sep_mask_cpu = (
                opt_sep_mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
            )
            optional_separator_pointer = ctypes.c_void_p(opt_sep_mask_cpu.data_ptr())

        initial_delta_cpu = (
            self.__initial_dp_score(
                log_b=log_b_cpu,
                dp_valid=dp_valid_cpu.bool(),
                opt_sep_mask=(None if opt_sep_mask_cpu is None else opt_sep_mask_cpu.bool()),
            )
            .to(dtype=torch.float32)
            .contiguous()
        )

        path_cpu = torch.empty(
            (B, T_speech),
            dtype=torch.int64,
            device="cpu",
        )
        c_viterbi_logp_cpu = torch.empty(
            (B,),
            dtype=torch.float32,
            device="cpu",
        )

        status = function(
            ctypes.c_void_p(log_b_cpu.data_ptr()),
            ctypes.c_void_p(dp_valid_cpu.data_ptr()),
            ctypes.c_void_p(initial_delta_cpu.data_ptr()),
            ctypes.c_void_p(spec_lengths_cpu.data_ptr()),
            ctypes.c_void_p(text_lengths_cpu.data_ptr()),
            optional_separator_pointer,
            ctypes.c_int64(B),
            ctypes.c_int64(T_speech),
            ctypes.c_int64(T_text),
            ctypes.c_float(float(self.neg_large)),
            ctypes.c_void_p(path_cpu.data_ptr()),
            ctypes.c_void_p(c_viterbi_logp_cpu.data_ptr()),
        )

        if status != 0:
            raise RuntimeError(f"viterbi_forward_backtrack_f32 failed with status={status}.")

        path = path_cpu.to(device=device)
        valid_t = path >= 0

        hard_attn = log_b.new_zeros((B, T_speech, T_text))

        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, T_speech)
        t_idx = torch.arange(T_speech, device=device).view(1, T_speech).expand(B, T_speech)
        j_idx = path.clamp_min(0)

        hard_attn[b_idx[valid_t], t_idx[valid_t], j_idx[valid_t]] = 1.0
        hard_attn = hard_attn.masked_fill(~dp_valid, 0.0)
        hard_durations = hard_attn.sum(dim=1)

        # Recompute the selected path score from log_b so viterbi_logp keeps
        # its gradient with respect to log_b. The C-produced value is used
        # only to execute and validate the C ABI output buffer.
        selected_log_b = torch.gather(
            log_b,
            dim=2,
            index=j_idx.unsqueeze(-1),
        ).squeeze(-1)
        viterbi_logp = selected_log_b.masked_fill(~valid_t, 0.0).sum(dim=1)

        return hard_attn, hard_durations, path, viterbi_logp

    def __viterbi_decode_python(
        self,
        log_b: torch.Tensor,
        dp_valid: torch.Tensor,
        spec_lengths: torch.Tensor,
        text_lengths: torch.Tensor,
        opt_sep_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original PyTorch Viterbi implementation used as a fallback."""
        B, T_speech, T_text = log_b.shape
        device = log_b.device

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
            dp_valid=dp_valid,
            opt_sep_mask=opt_sep_mask,
        )
        delta_steps.append(delta_t)

        j_idx = torch.arange(T_text, device=device).view(1, T_text).expand(B, -1)
        prev_stay = j_idx
        prev_adv = torch.clamp(j_idx - 1, min=0)

        if opt_sep_mask is None:
            candidate_prev = torch.stack([prev_stay, prev_adv], dim=0)
        else:
            prev_skip = torch.clamp(j_idx - 2, min=0)
            candidate_prev = torch.stack([prev_stay, prev_adv, prev_skip], dim=0)

        for t in range(1, T_speech):
            stay, adv, skip = self.__prev_to_current_scores(
                prev_score=delta_t,
                opt_sep_mask=opt_sep_mask,
            )
            if skip is None:
                candidate_scores = torch.stack([stay, adv], dim=0)
            else:
                candidate_scores = torch.stack([stay, adv, skip], dim=0)

            best_move = torch.argmax(candidate_scores, dim=0)
            best_prev_score = torch.gather(
                candidate_scores,
                dim=0,
                index=best_move.unsqueeze(0),
            ).squeeze(0)

            delta_t = log_b[:, t, :] + best_prev_score
            delta_t = delta_t.masked_fill(~dp_valid[:, t, :], self.neg_large)

            backptr[:, t, :] = torch.gather(
                candidate_prev,
                dim=0,
                index=best_move.unsqueeze(0),
            ).squeeze(0)

            delta_steps.append(delta_t)

        log_delta = torch.stack(delta_steps, dim=1)

        batch_idx = torch.arange(B, device=device)
        viterbi_logp = log_delta[batch_idx, spec_lengths - 1, text_lengths - 1]

        backptr_cpu = backptr.detach().cpu()
        spec_lengths_cpu = spec_lengths.detach().cpu()
        text_lengths_cpu = text_lengths.detach().cpu()

        path_cpu = torch.full(
            (B, T_speech),
            -1,
            dtype=torch.long,
        )

        for b in range(B):
            t_end = int(spec_lengths_cpu[b]) - 1
            j_cur = int(text_lengths_cpu[b]) - 1

            for t in range(t_end, -1, -1):
                path_cpu[b, t] = j_cur

                if t > 0:
                    j_cur = int(backptr_cpu[b, t, j_cur])

        path = path_cpu.to(device=device)

        hard_attn = log_b.new_zeros((B, T_speech, T_text))
        valid_t = path >= 0

        b_idx = torch.arange(B, device=device).view(B, 1).expand(B, T_speech)
        t_idx = torch.arange(T_speech, device=device).view(1, T_speech).expand(B, T_speech)
        j_idx = path.clamp_min(0)

        hard_attn[b_idx[valid_t], t_idx[valid_t], j_idx[valid_t]] = 1.0
        hard_attn = hard_attn.masked_fill(~dp_valid, 0.0)

        hard_durations = hard_attn.sum(dim=1)

        return hard_attn, hard_durations, path, viterbi_logp

    def _compute_log_node_potential(
        self,
        evidence: torch.Tensor,  # (B, T_s, T_t)
        unary_valid: torch.Tensor,  # (B, T_s, T_t)
    ) -> torch.Tensor:
        """
        Compute unary log potential log_b[t, j].

        Supports:
            raw:
                unnormalized raw logit score.

            bernoulli:
                raw logit score used as the path-dependent log-odds term
                of the full Bernoulli emission objective. The path-independent
                negative-cell penalty is added outside the DP.

            global:
                framewise text-axis log-softmax.
        """
        if self.unary_support_type in ("raw", "bernoulli"):
            temperature = max(float(self.unary_temperature), 1e-6)
            score = evidence / temperature
            return score.masked_fill(~unary_valid, self.neg_large)

        if self.unary_support_type == "global":
            return self.__log_softmax_normalization(
                evidence=evidence,
                unary_valid=unary_valid,
            )

        raise RuntimeError(f"Unexpected unary_support_type: {self.unary_support_type!r}")

    def __log_softmax_normalization(
        self,
        evidence: torch.Tensor,
        unary_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        classification for text-axis
        For each frame t, the denominator is computed over length-valid text states.
        Strict reachability is not applied during unary normalization.

            log_phi[t, j]
            =
            e[t, j] / tau
            -
            logsumexp_{k: length_valid(t,k)} e[t, k] / tau

        Padding-invalid states are masked after log_phi is computed.
        Strict DP reachability is applied later by forward-backward/Viterbi.
        """
        if self.unary_temperature <= 0:
            raise ValueError(
                f"Temperature must be positive for global-support log potential, got {self.unary_temperature}."
            )

        temperature = max(float(self.unary_temperature), 1e-6)

        score = evidence / temperature

        score_for_norm = score.masked_fill(~unary_valid, self.neg_large)

        log_denom = torch.logsumexp(
            score_for_norm,
            dim=-1,
            keepdim=True,
        )

        log_phi = score - log_denom
        log_phi = log_phi.masked_fill(~unary_valid, self.neg_large)

        return log_phi

    def __bernoulli_negative_penalty(
        self,
        evidence: torch.Tensor,
        penalty_valid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Path-independent negative-cell penalty for the full Bernoulli emission
        objective.

        Computes:

            sum_{(t,j) in D} softplus(s[t,j])

        where s[t,j] = evidence[t,j] / tau.

        This equals:

            - sum_{(t,j) in D} log sigmoid(-s[t,j])
        """
        if self.unary_temperature <= 0:
            raise ValueError(
                f"Temperature must be positive for Bernoulli penalty, got {self.unary_temperature}."
            )

        temperature = max(float(self.unary_temperature), 1e-6)
        score = evidence / temperature

        penalty = F.softplus(score)
        penalty = penalty.masked_fill(~penalty_valid, 0.0)

        return penalty.sum(dim=(1, 2))  # (B,)

    def __initial_dp_score(
        self,
        log_b: torch.Tensor,
        dp_valid: torch.Tensor,
        opt_sep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Initial DP score.

        Normally:
            z_0 = 0

        If token 0 is optional:
            z_0 in {0, 1}
        """
        B, _, T_text = log_b.shape

        score = log_b.new_full(
            (B, T_text),
            self.neg_large,
        )

        # Normal start from token 0.
        score[:, 0] = log_b[:, 0, 0]

        # Skip the first token and start from token 1.
        if opt_sep_mask is not None and T_text >= 2:
            can_skip_first = opt_sep_mask[:, 0]

            score[:, 1] = torch.where(
                can_skip_first,
                log_b[:, 0, 1],
                score[:, 1],
            )

        score = score.masked_fill(
            ~dp_valid[:, 0, :],
            self.neg_large,
        )

        return score

    def __prev_to_current_scores(
        self,
        prev_score: torch.Tensor,
        opt_sep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Candidate scores from previous frame/state to current frame/state.

        For current state j:
            stay = prev_score[j]
            adv  = prev_score[j - 1]

        This is the original strict monotone stay/advance topology.
        """
        B, T_text = prev_score.shape
        device = prev_score.device

        stay = prev_score

        adv = torch.cat(
            [
                prev_score.new_full((B, 1), self.neg_large),
                prev_score[:, :-1],
            ],
            dim=1,
        )

        if opt_sep_mask is None:
            return stay, adv, None

        skip = torch.cat(
            [
                prev_score.new_full((B, 2), self.neg_large),
                prev_score[:, :-2],
            ],
            dim=1,
        )

        skip_allowed = torch.zeros(
            (B, T_text),
            dtype=torch.bool,
            device=device,
        )

        if T_text > 2:
            # For current j, skip is allowed iff token j - 1 is optional.
            skip_allowed[:, 2:] = opt_sep_mask[:, 1:-1]

        skip = skip.masked_fill(~skip_allowed, self.neg_large)

        return stay, adv, skip

    def __current_to_next_scores(
        self,
        next_score: torch.Tensor,
        opt_sep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Candidate scores from current frame/state to next frame/state.

        For current state j:
            stay = next_score[j]
            adv  = next_score[j + 1]

        Here:
            next_score[:, k] = log_b[t + 1, k] + beta[t + 1, k]

        This is the original strict monotone stay/advance topology.
        """
        B, T_text = next_score.shape
        device = next_score.device

        stay = next_score

        adv = torch.cat(
            [
                next_score[:, 1:],
                next_score.new_full((B, 1), self.neg_large),
            ],
            dim=1,
        )

        if opt_sep_mask is None:
            return stay, adv, None

        skip = torch.cat(
            [
                next_score[:, 2:],
                next_score.new_full((B, 2), self.neg_large),
            ],
            dim=1,
        )

        skip_allowed = torch.zeros(
            (B, T_text),
            dtype=torch.bool,
            device=device,
        )

        if T_text > 2:
            # For current j, skip is allowed iff token j + 1 is optional.
            skip_allowed[:, :-2] = opt_sep_mask[:, 1:-1]

        skip = skip.masked_fill(~skip_allowed, self.neg_large)

        return stay, adv, skip

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
          2. final state N - 1 can still be reached by final frame T - 1.

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

    @staticmethod
    def _optional_separator_reachability_mask(
        spec_lengths: torch.Tensor,  # (B,)
        text_lengths: torch.Tensor,  # (B,)
        opt_sep_mask: torch.Tensor,  # (B, T_text)
        T_speech: int,
        T_text: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Reachability mask for monotone paths with optional separator skips.

        Allowed transitions:
            j -> j      stay
            j -> j + 1  advance
            j -> j + 2  skip token j + 1, iff that token is optional

        Initial states:
            z_0 = 0
            z_0 = 1, iff token 0 is optional

        The terminal state remains fixed to text_lengths[b] - 1.
        """
        B = spec_lengths.size(0)
        inf = torch.iinfo(torch.long).max // 4

        min_prefix = torch.full(
            (B, T_text),
            fill_value=inf,
            dtype=torch.long,
            device=device,
        )

        min_prefix[:, 0] = 1

        if T_text >= 2:
            min_prefix[:, 1] = torch.where(
                opt_sep_mask[:, 0],
                torch.ones(B, dtype=torch.long, device=device),
                torch.full(
                    (B,),
                    2,
                    dtype=torch.long,
                    device=device,
                ),
            )

        for j in range(2, T_text):
            # Normal advance:
            normal_cost = min_prefix[:, j - 1] + 1

            # Optional skip:
            skip_cost = min_prefix[:, j - 2] + 1
            skip_allowed = opt_sep_mask[:, j - 1]

            min_prefix[:, j] = torch.where(
                skip_allowed,
                torch.minimum(normal_cost, skip_cost),
                normal_cost,
            )

        min_suffix = torch.full(
            (B, T_text),
            fill_value=inf,
            dtype=torch.long,
            device=device,
        )

        batch_idx = torch.arange(B, device=device)
        final_j = text_lengths - 1

        min_suffix[batch_idx, final_j] = 1

        for j in range(T_text - 2, -1, -1):
            update_mask = j < final_j

            # Normal advance:
            normal_cost = min_suffix[:, j + 1] + 1
            best_cost = normal_cost

            # Optional skip:
            if j + 2 < T_text:
                skip_cost = min_suffix[:, j + 2] + 1
                skip_allowed = opt_sep_mask[:, j + 1]

                best_cost = torch.where(
                    skip_allowed,
                    torch.minimum(normal_cost, skip_cost),
                    normal_cost,
                )

            min_suffix[:, j] = torch.where(
                update_mask,
                best_cost,
                min_suffix[:, j],
            )

        t = torch.arange(
            T_speech,
            device=device,
        ).view(1, T_speech, 1)

        j = torch.arange(
            T_text,
            device=device,
        ).view(1, 1, T_text)

        T = spec_lengths.view(B, 1, 1)
        N = text_lengths.view(B, 1, 1)

        within_lengths = (t < T) & (j < N)

        reachable_from_start = (min_prefix.view(B, 1, T_text) - 1) <= t

        completable_to_end = (min_suffix.view(B, 1, T_text) - 1) <= (T - 1 - t)

        return within_lengths & reachable_from_start & completable_to_end
