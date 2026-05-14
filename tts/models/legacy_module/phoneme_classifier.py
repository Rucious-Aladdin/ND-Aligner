from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhonemeClassifier(nn.Module):
    """
    Token/phoneme classifier from text latent features.

    Args:
        in_dim:
            Channel dimension of z_text.
        n_vocab:
            Number of tokenizer symbols/classes.
        hidden_dim:
            Hidden dimension for classifier head.
        dropout_p:
            Dropout probability.

    Input:
        z_text:    (B, C, T_text)
        text_mask: (B, T_text) or (B, 1, T_text)

    Output:
        logits: (B, n_vocab, T_text)
    """

    def __init__(
        self,
        in_dim: int,
        n_vocab: int,
        hidden_dim: int | None = None,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        self.in_dim = in_dim
        self.n_vocab = n_vocab
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Conv1d(hidden_dim, n_vocab, kernel_size=1),
        )

    @staticmethod
    def _make_mask(
        text_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Returns:
            mask: (B, 1, T_text)
        """
        if text_mask.dim() == 2:
            text_mask = text_mask.unsqueeze(1)

        return text_mask.to(dtype=dtype)

    @override
    def forward(
        self,
        z_text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_text:    (B, C, T_text)
            text_mask: (B, T_text) or (B, 1, T_text)

        Returns:
            logits: (B, n_vocab, T_text)
        """
        mask = self._make_mask(text_mask, dtype=z_text.dtype)

        z_text = z_text.detach() * mask
        logits = self.net(z_text)

        # Invalid positions are not used in the masked CE loss anyway.
        # This masking is mostly for safer logging/debugging.
        logits = logits * mask

        return logits

    def compute_loss(
        self,
        z_text: torch.Tensor,
        text_mask: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Masked token classification loss and accuracy.

        Args:
            z_text:    (B, C, T_text)
            text_mask: (B, T_text) or (B, 1, T_text)
            token_ids: (B, T_text), LongTensor

        Returns:
            loss: scalar cross-entropy loss
            acc : scalar masked accuracy
        """
        logits = self.forward(z_text=z_text, text_mask=text_mask)

        if text_mask.dim() == 3:
            mask = text_mask.squeeze(1).bool()
        else:
            mask = text_mask.bool()

        # logits: (B, V, T) -> (B, T, V)
        logits_bt = logits.transpose(1, 2).contiguous()

        loss = F.cross_entropy(
            logits_bt.reshape(-1, self.n_vocab),
            token_ids.contiguous().reshape(-1),
            reduction="none",
        )

        loss = loss.view_as(token_ids)
        loss = loss.masked_fill(~mask, 0.0)

        denom = mask.sum().float().clamp_min(1.0)
        loss = loss.sum() / denom

        with torch.no_grad():
            pred = logits.argmax(dim=1)  # (B, T_text)
            correct = (pred == token_ids) & mask
            acc = correct.float().sum() / denom

        return loss, acc
