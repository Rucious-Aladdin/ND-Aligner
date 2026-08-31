from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import torch

TokenIdInput = torch.Tensor | np.ndarray | Sequence[int] | Sequence[Sequence[int]]
EncodedTokenSequence = np.ndarray | Sequence[int]
DecodedText = str | list[str]


class BaseTokenizer(ABC):
    """
    Abstract base class for text/phoneme tokenizers.

    Required behavior:
        tokenizer(text) -> LongTensor with shape (1, T)
        tokenizer.encode(text) -> 1D token-id sequence
        tokenizer.decode(token_ids) -> decoded string or list of strings
        tokenizer.n_vocab -> vocabulary size
        tokenizer.vocab_labels -> id-to-label list
    """

    def __call__(self, text: str) -> torch.Tensor:
        """
        Tokenize raw text.

        Returns:
            LongTensor with shape (1, T).
        """
        sequence = self.encode(text)
        return torch.as_tensor(sequence, dtype=torch.long).unsqueeze(0)

    def decode_to_symbols(self, token_ids: TokenIdInput) -> list[str] | list[list[str]]:
        """
        Decode token ids into a token-level symbol list.

        Unlike decode(), this must preserve 1-to-1 alignment with token ids.
        """
        decoded = self.decode(token_ids)

        if isinstance(decoded, str):
            return list(decoded)

        return [list(item) for item in decoded]

    @abstractmethod
    def encode(self, text: str) -> EncodedTokenSequence:
        """
        Encode raw text into a 1D token-id sequence.

        Returns:
            A 1D sequence of integer token ids.
        """
        raise NotImplementedError()

    @abstractmethod
    def decode(self, token_ids: TokenIdInput) -> DecodedText:
        """
        Decode token ids back into a readable token/string representation.

        Args:
            token_ids:
                1D or 2D Tensor / ndarray / sequence.

        Returns:
            str for 1D or batch-size-1 input, otherwise list[str].
        """
        raise NotImplementedError()

    @abstractmethod
    def to_token_string(self, text: str) -> str:
        """
        Convert raw text into a human-readable token string for debugging.

        Example:
            IPA tokenizer:
                "Please call Stella." -> "<sp> pliːz ... <sp>"
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_vocab(self) -> int:
        """
        Vocabulary size used for text embedding.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def vocab_labels(self) -> list[str]:
        """
        Vocabulary id-to-label list.

        Must satisfy:
            len(tokenizer.vocab_labels) == tokenizer.n_vocab
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def seperator_id(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def ignore_symbols(self) -> set[str]:
        raise NotImplementedError()

    @property
    def span_only_symbols(self) -> set[str]:
        """
        Symbols that occupy a token position inside a word but contribute no
        character to it.

        Code that aligns decoded symbols against reference words must advance
        its token span across these symbols while leaving the match key
        unchanged, which is neither what ignore_symbols asks for (those are
        skipped entirely) nor what an ordinary symbol asks for.

        Defaults to the empty set; tokenizers that emit such symbols override
        this.
        """
        return set()
