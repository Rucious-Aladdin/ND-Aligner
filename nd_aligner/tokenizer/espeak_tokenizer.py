from __future__ import annotations

# IPA Phonemizer: https://github.com/bootphon/phonemizer
from collections.abc import Callable, Iterable, Sequence
from functools import cached_property
from typing import cast, override

import numpy as np
import phonemizer
import torch
from nltk.tokenize import word_tokenize

from .base_tokenizer import BaseTokenizer, TokenIdInput
from .espeak_letters import IGNORE_SYMBOLS, SYMBOL_DICTS


class TextCleaner:
    def __init__(self, dummy: None = None):
        self.word_index_dictionary = SYMBOL_DICTS
        self.index_word_dictionary = {idx: sym for sym, idx in self.word_index_dictionary.items()}

    def __call__(self, text: str):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                # print(text)
                pass
        return indexes

    def decode(self, token_ids: torch.Tensor | list[int]) -> str:
        """
        Decode token ids back to raw IPA symbol sequence.
        Args:
            token_ids: (T,) LongTensor or list[int]

        Returns:
            Raw IPA sequence string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long().tolist()

        return "".join(self.index_word_dictionary.get(int(idx), "") for idx in token_ids)


"""
Closure tokens for stop consonants.

A stop consonant is realized as a silent closure followed by a release burst,
and reference annotations such as TIMIT label the two separately. The
tokenizers here can therefore emit an explicit closure token before every stop,
so that a monotone aligner is free to spend frames on the closure rather than
folding it into the preceding phone.

This module holds only the vocabulary constants and the id-level transform;
the tokenizer that applies it is ESPEAKTokenizer, whose
`expand_closure_tokens` flag turns the transform on.
"""


CL_UNVOICED_ID = 178
CL_UNVOICED_SYMBOL = "<clu>"
UNVOICED_STOPS = frozenset({"p", "t", "k"})

CL_VOICED_ID = 179
CL_VOICED_SYMBOL = "<clv>"
VOICED_STOPS = frozenset({"b", "d", "ɡ"})

CLOSURE_IDS = frozenset({CL_UNVOICED_ID, CL_VOICED_ID})
CLOSURE_SYMBOLS = frozenset({CL_UNVOICED_SYMBOL, CL_VOICED_SYMBOL})


def closure_symbol_of(token_id: int) -> str | None:
    """
    Return the closure symbol for a token id, or None if it is not a closure.
    """
    if token_id == CL_UNVOICED_ID:
        return CL_UNVOICED_SYMBOL

    if token_id == CL_VOICED_ID:
        return CL_VOICED_SYMBOL

    return None


def expand_closures(
    token_ids: Iterable[int],
    symbol_of: Callable[[int], str],
) -> list[int]:
    """
    Insert a closure token before every stop consonant.

    Each stop is preceded by the unvoiced or voiced closure id according to the
    set its symbol belongs to; every other id passes through unchanged.

    Args:
        token_ids:
            The id sequence to transform.
        symbol_of:
            Maps an id to its symbol, so that this function does not need a
            tokenizer instance.

    Returns:
        The expanded id sequence.
    """
    out: list[int] = []

    for token_id in token_ids:
        symbol = symbol_of(token_id)

        if symbol in UNVOICED_STOPS:
            out.append(CL_UNVOICED_ID)
        elif symbol in VOICED_STOPS:
            out.append(CL_VOICED_ID)

        out.append(token_id)

    return out


class ESPEAKTokenizer(BaseTokenizer):
    def __init__(self, expand_closure_tokens: bool = False):
        """
        Args:
            expand_closure_tokens:
                Emit an explicit closure token before every stop consonant.
                See `closure_tokenizer` for what these are and why they exist.
        """
        self.cleaner = TextCleaner()
        self.phonemizer = (
            phonemizer.backend.EspeakBackend(  # pyright: ignore[reportAttributeAccessIssue]
                language="en-us",
                preserve_punctuation=True,
                with_stress=True,
            )
        )
        self.blank = " "
        self.expand_closure_tokens = expand_closure_tokens

    @override
    def encode(self, text: str) -> list[int]:
        ps = self.to_ipa(text)
        tokens = [
            SYMBOL_DICTS[self.blank],
        ] + self.cleaner(ps)
        tokens.append(SYMBOL_DICTS[self.blank])

        if self.expand_closure_tokens:
            tokens = expand_closures(tokens, self._symbol_of)

        return tokens

    @override
    def __call__(self, text: str) -> torch.Tensor:
        return super().__call__(text)

    def to_ipa(self, text: str) -> str:
        text = text.strip()
        text = text.replace('"', "")
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        return ps

    @override
    def to_token_string(self, text: str) -> str:
        return self.to_ipa(text)

    @override
    def decode_to_symbols(self, token_ids: TokenIdInput) -> list[str] | list[list[str]]:
        """
        Decode token ids into a token-level symbol list, one label per id.

        Unlike decode(), this keeps symbols that carry no IPA character, such
        as closure tokens, so that the result stays aligned with the input
        sequence. Plotting and any per-token analysis should use this.
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

        Args:
            token_ids:
                (T,), (1, T), or (B, T) Tensor / ndarray / list.

        Returns:
            str if input is 1D or batch size 1, otherwise list[str].
        """
        rows = self._to_id_rows(token_ids)
        decoded = ["".join(self._symbol_of(i) for i in row if i not in CLOSURE_IDS) for row in rows]

        if len(decoded) == 1:
            return decoded[0]

        return decoded

    @property
    @override
    def n_vocab(self) -> int:
        base = max(self.cleaner.word_index_dictionary.values()) + 1

        if self.expand_closure_tokens:
            return max(base, max(CLOSURE_IDS) + 1)

        return base

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

    @property
    @override
    def ignore_symbols(self):
        return IGNORE_SYMBOLS

    @property
    @override
    def span_only_symbols(self) -> set[str]:
        """
        Closure symbols occupy a token position but spell no IPA character, so
        a word span must cover them without their affecting the match key.
        """
        if self.expand_closure_tokens:
            return set(CLOSURE_SYMBOLS)

        return set()

    @property
    @override
    def seperator_id(self):
        blank = " "
        return self.cleaner.word_index_dictionary[blank]

    def _symbol_of(self, token_id: int) -> str:
        symbol = closure_symbol_of(token_id)

        if symbol is not None:
            return symbol

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
