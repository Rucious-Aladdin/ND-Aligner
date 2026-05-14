from typing import override

import torch
import torch.nn as nn


class TextEmbedder(nn.Module):
    """
    Minimal text embedder.

    Input:
        text_token_ids : token ids, (B, T_text)
        text_mask      : (B, T_text) or (B, 1, T_text)

    Output:
        h_text         : token embedding sequence, (B, hidden_channels, T_text)
    """

    def __init__(
        self,
        n_vocab: int,
        dim_out: int,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.hidden_channels = dim_out

        self.out = nn.Embedding(n_vocab, dim_out)
        nn.init.normal_(self.out.weight, mean=0.0, std=dim_out**-0.5)

    @override
    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_token_ids: (B, T_text)
            text_mask:      (B, T_text) or (B, 1, T_text)

        Returns:
            h_text:         (B, hidden_channels, T_text)
        """
        if text_mask.dim() == 2:
            text_mask = text_mask.unsqueeze(1)

        text_mask = text_mask.to(dtype=self.out.weight.dtype)

        # (B, T_text) -> (B, T_text, hidden_channels)
        h_text = self.out(text_token_ids)

        # (B, T_text, hidden_channels) -> (B, hidden_channels, T_text)
        h_text = h_text.transpose(1, 2)

        h_text = h_text * text_mask

        return h_text
