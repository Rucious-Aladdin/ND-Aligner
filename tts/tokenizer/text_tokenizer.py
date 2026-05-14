# IPA Phonemizer: https://github.com/bootphon/phonemizer
import phonemizer
import torch
from nltk.tokenize import word_tokenize

from .letters import SYMBOL_DICTS, EOS, BOS
from functools import cached_property


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
        BOS/EOS are preserved if they are included in token_ids.

        Args:
            token_ids: (T,) LongTensor or list[int]

        Returns:
            Raw IPA sequence string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long().tolist()

        return "".join(self.index_word_dictionary.get(int(idx), "") for idx in token_ids)


class TextTokenizer:
    def __init__(self):
        self.cleaner = TextCleaner()
        self.phonemizer = (
            phonemizer.backend.EspeakBackend(  # pyright: ignore[reportAttributeAccessIssue]
                language="en-us",
                preserve_punctuation=True,
                with_stress=True,
            )
        )

    def __call__(
        self,
        text: str,
    ) -> torch.Tensor:
        ps = self.to_ipa(text)
        tokens = self.cleaner(ps)
        tokens.insert(0, SYMBOL_DICTS[BOS])
        tokens.append(SYMBOL_DICTS[EOS])
        tokens = torch.LongTensor(tokens).unsqueeze(0)
        return tokens

    def to_ipa(self, text: str) -> str:
        text = text.strip()
        text = text.replace('"', "")
        ps = self.phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = " ".join(ps)
        return ps

    def decode(self, token_ids: torch.Tensor) -> str | list[str]:
        """
        Decode LongTensor token ids to raw IPA sequence.

        Args:
            token_ids:
                (T,) or (1, T) or (B, T) LongTensor.

        Returns:
            str if input is 1D or batch size 1, otherwise list[str].
        """
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

    @property
    def n_vocab(self):
        return max(self.cleaner.word_index_dictionary.values()) + 1

    @cached_property
    def vocab_labels(self) -> list[str]:
        """
        Decode all vocab ids into IPA/BOS/EOS labels.
        Empty or unknown ids are displayed as [idx].
        """
        labels: list[str] = []

        for idx in range(self.n_vocab):
            label = self.cleaner.decode([idx])
            if label == "":
                label = f"[{idx}]"
            labels.append(label)

        return labels
