from collections.abc import Sequence
from functools import cached_property
from typing import cast, override

import numpy as np
import torch

from .base_tokenizer import TokenIdInput
from .espeak_tokenizer import ESPEAKTokenizer

CL_UNVOICED_ID = 178
CL_UNVOICED_SYMBOL = "<clu>"
UNVOICED_STOPS = frozenset({"p", "t", "k"})

CL_VOICED_ID = 179
CL_VOICED_SYMBOL = "<clv>"
VOICED_STOPS = frozenset({"b", "d", "ɡ"})

CLOSURE_IDS = frozenset({CL_UNVOICED_ID, CL_VOICED_ID})
CLOSURE_SYMBOLS = frozenset({CL_UNVOICED_SYMBOL, CL_VOICED_SYMBOL})


class ClosureESPEAKTokenizer(ESPEAKTokenizer):
    @override
    def encode(self, text: str) -> list[int]:
        out: list[int] = []

        for token_id in super().encode(text):
            symbol = self._symbol_of(token_id)

            if symbol in UNVOICED_STOPS:
                out.append(CL_UNVOICED_ID)
            elif symbol in VOICED_STOPS:
                out.append(CL_VOICED_ID)

            out.append(token_id)

        return out

    @override
    def decode_to_symbols(self, token_ids: TokenIdInput) -> list[str] | list[list[str]]:
        """
        Decode token ids into a token-level symbol list, including closure symbols.
        """
        rows = self._to_id_rows(token_ids)
        decoded = [[self._symbol_of(i) for i in row] for row in rows]

        if len(decoded) == 1:
            return decoded[0]

        return decoded

    @override
    def decode(self, token_ids: TokenIdInput) -> str | list[str]:
        """
        Decode token ids to raw IPA symbol sequence, dropping closure symbols.
        """
        rows = self._to_id_rows(token_ids)
        decoded = ["".join(self._symbol_of(i) for i in row if i not in CLOSURE_IDS) for row in rows]

        if len(decoded) == 1:
            return decoded[0]

        return decoded

    @property
    @override
    def n_vocab(self) -> int:
        return max(super().n_vocab, max(CLOSURE_IDS) + 1)

    @cached_property
    @override
    def vocab_labels(self) -> list[str]:
        labels: list[str] = []

        for idx in range(self.n_vocab):
            label = self._symbol_of(idx)
            if label == "":
                label = f"[{idx}]"
            labels.append(label)

        return labels

    def _symbol_of(self, token_id: int) -> str:
        if token_id == CL_UNVOICED_ID:
            return CL_UNVOICED_SYMBOL
        if token_id == CL_VOICED_ID:
            return CL_VOICED_SYMBOL

        return self.cleaner.index_word_dictionary.get(token_id, "")

    @staticmethod
    def _to_id_rows(token_ids: TokenIdInput) -> list[list[int]]:
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.from_numpy(token_ids)

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long()

            if token_ids.ndim == 1:
                return [token_ids.tolist()]
            if token_ids.ndim == 2:
                return [row.tolist() for row in token_ids]

            raise ValueError(
                f"expected token_ids with shape (T,) or (B, T), got {tuple(token_ids.shape)}."
            )

        seq = list(token_ids)

        if len(seq) > 0 and isinstance(seq[0], Sequence):
            return [list(cast(Sequence[int], row)) for row in seq]

        return [[cast(int, i) for i in seq]]
