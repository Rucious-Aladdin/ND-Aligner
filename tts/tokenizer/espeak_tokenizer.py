# IPA Phonemizer: https://github.com/bootphon/phonemizer
from functools import cached_property
from typing import override

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


class ESPEAKTokenizer(BaseTokenizer):
    def __init__(self):
        self.cleaner = TextCleaner()
        self.phonemizer = (
            phonemizer.backend.EspeakBackend(  # pyright: ignore[reportAttributeAccessIssue]
                language="en-us",
                preserve_punctuation=True,
                with_stress=True,
            )
        )
        self.blank = " "

    @override
    def encode(self, text: str) -> list[int]:
        ps = self.to_ipa(text)
        tokens = [
            SYMBOL_DICTS[self.blank],
        ] + self.cleaner(ps)
        tokens.append(SYMBOL_DICTS[self.blank])
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
    def decode(self, token_ids: TokenIdInput) -> str | list[str]:
        """
        Decode token ids to raw IPA symbol sequence.

        Args:
            token_ids:
                (T,), (1, T), or (B, T) Tensor / ndarray / list.

        Returns:
            str if input is 1D or batch size 1, otherwise list[str].
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = torch.from_numpy(token_ids)

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long()

            if token_ids.dim() == 1:
                return self.cleaner.decode(token_ids)

            if token_ids.dim() != 2:
                raise ValueError(
                    f"decode expects token_ids with shape (T,) or (B, T), "
                    + f"but got {tuple(token_ids.shape)}."
                )

            decoded = [self.cleaner.decode(row) for row in token_ids]

            if len(decoded) == 1:
                return decoded[0]

            return decoded

        token_ids = list(token_ids)  # type: ignore

        if len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
            decoded = [self.cleaner.decode(list(row)) for row in token_ids]  # type: ignore[arg-type]

            if len(decoded) == 1:
                return decoded[0]

            return decoded

        return self.cleaner.decode([int(idx) for idx in token_ids])  # type: ignore[arg-type]

    @property
    @override
    def n_vocab(self) -> int:
        return max(self.cleaner.word_index_dictionary.values()) + 1

    @cached_property
    @override
    def vocab_labels(self) -> list[str]:
        labels: list[str] = []

        for idx in range(self.n_vocab):
            label = self.cleaner.decode([idx])
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
    def seperator_id(self):
        blank = " "
        return self.cleaner.word_index_dictionary[blank]
