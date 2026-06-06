import math
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.convolution import ConformerConvModule
from ..layers.feed_forward import FeedForwardModule
from ..layers.linear import Linear
from ..layers.residual_connection import ResidualConnectionModule
from ..utils.positional_encoding import RelPositionalEncoding


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = Linear(d_model, d_model)
        self.key_proj = Linear(d_model, d_model)
        self.value_proj = Linear(d_model, d_model)
        self.pos_proj = Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        # >>> GATED ATTENTION (Qwen3 Style): Added gate projection
        # This helps mitigate the 'attention sink' problem by allowing the model
        # to gate the attention output based on the input query.
        self.gate_proj = Linear(d_model, d_model)
        # <<< GATED ATTENTION END

        self.out_proj = Linear(d_model, d_model)

    @override
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_embedding: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = value.size(0)
        query_orig = query

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = (
            self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        )
        value = (
            self.value_proj(value)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        pos_embedding = self.pos_proj(pos_embedding).view(
            batch_size, -1, self.num_heads, self.d_head
        )

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul(
            (query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1)
        )
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        # gate_score is computed from the original query input.
        gate = torch.sigmoid(self.gate_proj(query_orig))
        context = context * gate

        return self.out_proj(context)

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1
        )
        pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]

        return pos_score


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.1,
        attn_window_size: int = 3,
    ):
        super().__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.attn_window_size = attn_window_size

    @override
    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None):
        batch_size, seq_len, _ = inputs.size()
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)

        # Sliding Window Mask
        if self.attn_window_size > 0:
            # Create a distance matrix: [seq_len, seq_len]
            # dist[i, j] = |i - j|
            idx = torch.arange(seq_len, device=inputs.device)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            window_mask = dist > self.attn_window_size
            window_mask = window_mask.unsqueeze(0)  # [1, T, T]

            if mask is not None:
                # Combine with padding mask (OR operation)
                # mask is (B, 1, T) or (B, T, T)
                mask = mask | window_mask
            else:
                mask = window_mask

        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class ConformerBlock(nn.Module):
    """
    Conformer block contains two Feed Forward modules sandwiching the Multi-Headed Self-Attention module
    and the Convolution module. This sandwich structure is inspired by Macaron-Net, which proposes replacing
    the original feed-forward layer in the Transformer block into two half-step feed-forward layers,
    one before the attention layer and one after.

    Args:
        encoder_dim (int, optional): Dimension of conformer encoder
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
        cond_in_channels (int): dimension of conditions (e.g., speker-embeddings)
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, time, dim): Tensor produces by conformer block.
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        cond_in_channels: int = 0,
        attn_window_size: int = 3,
    ):
        super().__init__()
        if half_step_residual:
            self.feed_forward_residual_factor = 0.5
        else:
            self.feed_forward_residual_factor = 1

        self.use_cond = cond_in_channels > 0
        if self.use_cond:
            self.cond_proj = nn.Linear(cond_in_channels, encoder_dim)

        self.ff1 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )
        self.attn = ResidualConnectionModule(
            module=MultiHeadedSelfAttentionModule(
                d_model=encoder_dim,
                num_heads=num_attention_heads,
                dropout_p=attention_dropout_p,
                attn_window_size=attn_window_size,
            ),
        )
        self.conv = ResidualConnectionModule(
            module=ConformerConvModule(
                in_channels=encoder_dim,
                kernel_size=conv_kernel_size,
                expansion_factor=conv_expansion_factor,
                dropout_p=conv_dropout_p,
            ),
        )
        self.ff2 = ResidualConnectionModule(
            module=FeedForwardModule(
                encoder_dim=encoder_dim,
                expansion_factor=feed_forward_expansion_factor,
                dropout_p=feed_forward_dropout_p,
            ),
            module_factor=self.feed_forward_residual_factor,
        )
        self.layer_norm = nn.LayerNorm(encoder_dim)

    @override
    def forward(
        self,
        inputs: torch.Tensor,
        cond: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,  # (B, T)
    ) -> torch.Tensor:

        x = inputs

        if self.use_cond and cond is not None:
            condition = self.cond_proj(cond).unsqueeze(1)
        else:
            condition = 0

        # Apply mask to input to prevent zero-padding influence
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        x = x + condition
        x = self.ff1(x)

        if mask is not None:
            attn_mask = (mask == 0).unsqueeze(1)
        else:
            attn_mask = None

        x = self.attn(x, mask=attn_mask)
        x = self.conv(x)
        x = self.ff2(x)

        # Removed redundant layer_norm here to maintain feature scale stability
        # The internal modules already handle normalization.

        if mask is not None:
            x = x * mask.unsqueeze(-1).float()

        return x


class ConformerEncoder(nn.Module):
    """
    Conformer encoder processes the length-regulated input with a linear projection layer
    and then with a number of conformer blocks.

    Args:
        input_dim (int, optional): Dimension of input vector (e.g., text hidden dim).
        encoder_dim (int, optional): Dimension of conformer encoder.
        num_layers (int, optional): Number of conformer blocks (Default 6 for TTS).
        num_attention_heads (int, optional): Number of attention heads.
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module.
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module.
        input_dropout_p (float, optional): Probability of input projection dropout.
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout.
        attention_dropout_p (float, optional): Probability of attention module dropout.
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout.
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel.
        half_step_residual (bool): Flag indication whether to use half step residual or not.
        cond_in_channels (int): dimension of conditions (e.g., speaker-embeddings).

    Inputs: inputs, input_lengths, cond
        - **inputs** (batch, time, dim): Tensor containing input vector.
        - **input_lengths** (batch): list of sequence input lengths.
        - **cond** (batch, cond_in_channels): Tensor containing condition vectors.

    Returns: outputs, output_lengths
        - **outputs** (batch, time, encoder_dim): Tensor produced by conformer encoder.
        - **output_lengths** (batch): list of sequence output lengths.
    """

    def __init__(
        self,
        input_dim: int = 80,
        encoder_dim: int = 512,
        cond_in_channels: int = 192,
        num_layers: int = 6,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = True,
        attn_window_size: int = 3,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            Linear(input_dim, encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    encoder_dim=encoder_dim,
                    num_attention_heads=num_attention_heads,
                    feed_forward_expansion_factor=feed_forward_expansion_factor,
                    conv_expansion_factor=conv_expansion_factor,
                    feed_forward_dropout_p=feed_forward_dropout_p,
                    attention_dropout_p=attention_dropout_p,
                    conv_dropout_p=conv_dropout_p,
                    conv_kernel_size=conv_kernel_size,
                    half_step_residual=half_step_residual,
                    cond_in_channels=cond_in_channels,
                    attn_window_size=attn_window_size,
                )
                for _ in range(num_layers)
            ]
        )

    def count_parameters(self) -> int:
        """Count parameters of encoder"""
        return sum([p.numel() for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        """Update dropout probability of encoder"""
        for _, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    @override
    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        cond: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagate an `inputs` for encoder training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            cond (torch.FloatTensor | None): Speaker or emotion embedding.

        Returns:
            (Tensor, Tensor)

            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, encoder_dim)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        outputs = self.input_projection(inputs)

        if mask is not None:
            outputs = outputs * mask.unsqueeze(-1).float()

        output_lengths = input_lengths

        for layer in self.layers:
            outputs = layer(outputs, cond=cond, mask=mask)

        return outputs, output_lengths
