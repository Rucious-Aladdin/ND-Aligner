import re
from functools import cached_property
from pathlib import Path
from string import punctuation
from typing import ClassVar, override

import nltk
import numpy as np
import torch
from g2p_en import G2p

from tts.baseline.FastSpeech2.text import sequence_to_text, text_to_sequence
from tts.baseline.FastSpeech2.text.cleaners import english_cleaners
from tts.baseline.FastSpeech2.text.symbols import symbols

from .base_tokenizer import BaseTokenizer, TokenIdInput

# nltk.download("averaged_perceptron_tagger_eng")


class ARPATokenizer(BaseTokenizer):
    """
    English ARPAbet tokenizer with optional special tokens. (FastSpeech2-compatible)

    Special token ids are appended after the original FastSpeech2 symbol table:
        <BLANK> -> len(symbols) + 2
        <UNK>   -> len(symbols) + 3

    Therefore, the downstream text embedding size must be tokenizer.n_vocab.
    """

    BLANK: ClassVar[str] = "<BLANK>"
    UNK: ClassVar[str] = "<UNK>"

    def __init__(
        self,
        lexicon_path: str | Path | None = None,
        add_blank_between_words: bool = True,
    ):
        self.lexicon_path = Path(lexicon_path) if lexicon_path is not None else None
        self.add_blank_between_words = add_blank_between_words

        if self.lexicon_path is not None:
            self.lexicon = self._read_lexicon(self.lexicon_path)
        else:
            self.lexicon = {}

        self.g2p = G2p()

        self.base_vocab_size = len(symbols)

        self.base_vocab_size = len(symbols)
        self.symbol_set = set(symbols)

        self.bos_id = self.base_vocab_size
        self.eos_id = self.base_vocab_size + 1
        self.blank_id = self.base_vocab_size + 2
        self.unk_id = self.base_vocab_size + 3

        self._ignore_symbols: set[str] = {
            self.BLANK,
            self.UNK,
        }

        self.special_token_to_id = {
            self.BLANK: self.blank_id,
            self.UNK: self.unk_id,
        }
        self.special_id_to_token = {idx: token for token, idx in self.special_token_to_id.items()}

    def to_phone_tokens(
        self,
        text: str,
    ) -> list[str]:
        """
        Convert raw text into a token-level phone sequence.

        Example:
            "Please call Stella."
            ->
            ["<BLANK>", "P", "L", "IY1", "Z", "<BLANK>",
             "K", "AO1", "L", "<BLANK>", "S", "T", "EH1", "L", "AH0", "<BLANK>"]
        """

        words = self._split_words(text)

        phone_tokens: list[str] = []

        for word_idx, word in enumerate(words):
            if word_idx > 0 and self.add_blank_between_words:
                phone_tokens.append(self.BLANK)

            phone_tokens.extend(self._word_to_phones(word))
        phone_tokens.insert(0, self.BLANK)
        phone_tokens.append(self.BLANK)

        return phone_tokens

    @override
    def to_token_string(self, text: str) -> str:
        """
        Human-readable token string for word mapping/debugging.
        BLANK is excluded from the string because word-span keys are built by
        concatenating token chunks.
        """
        tokens = self.to_phone_tokens(text)

        tokens = [
            token
            for token in tokens
            if token
            not in {
                self.BLANK,
                self.UNK,
            }
        ]

        return " ".join(tokens)

    @override
    def encode(self, text: str) -> np.ndarray:
        """
        Encode raw text into token ids.

        Returns:
            np.ndarray, shape (T,)

        The returned sequence may contain ids beyond len(symbols) when <BLANK> is enabled.
        """
        phone_tokens = self.to_phone_tokens(text)

        sequence: list[int] = []
        phone_chunk: list[str] = []

        def flush_phone_chunk() -> None:
            nonlocal phone_chunk, sequence

            if len(phone_chunk) == 0:
                return

            sequence.extend(self._encode_phone_chunk(phone_chunk, source_text=text))
            phone_chunk = []

        for token in phone_tokens:
            if token in self.special_token_to_id:
                flush_phone_chunk()
                sequence.append(self.special_token_to_id[token])
            else:
                phone_chunk.append(token)

        flush_phone_chunk()

        return np.array(sequence, dtype=np.int64)

    @override
    def __call__(self, text: str) -> torch.Tensor:
        return super().__call__(text)

    @override
    def decode(self, token_ids: TokenIdInput) -> str | list[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.astype(np.int64).tolist()

        if len(token_ids) > 0 and isinstance(token_ids[0], list):
            raise ValueError(
                "decode expects a 1D token sequence, "
                + f"but got nested sequence with length {len(token_ids)}."
            )

        token_ids = [int(x) for x in token_ids]  # type: ignore

        pieces: list[str] = []
        normal_ids: list[int] = []

        def flush_normal_ids() -> None:
            nonlocal normal_ids, pieces

            if len(normal_ids) == 0:
                return

            pieces.append(sequence_to_text(normal_ids))
            normal_ids = []

        for idx in token_ids:
            if idx in self.special_id_to_token:
                flush_normal_ids()
                pieces.append(self.special_id_to_token[idx])
            else:
                normal_ids.append(idx)

        flush_normal_ids()

        return " ".join(piece for piece in pieces if piece)

    @override
    def decode_to_symbols(self, token_ids: TokenIdInput) -> list[str] | list[list[str]]:
        """
        Decode token ids into token-level symbols.

        This preserves:

            len(output) == len(token_ids)

        Important:
            <BLANK> is returned as literal space " " so WordsMapper treats it
            as a word boundary.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().long().tolist()
        elif isinstance(token_ids, np.ndarray):
            token_ids = token_ids.astype(np.int64).tolist()

        if len(token_ids) > 0 and isinstance(token_ids[0], list):
            return [self.decode_to_symbols(row) for row in token_ids]  # type: ignore[arg-type]

        ids = [int(x) for x in token_ids]  # type: ignore[arg-type]

        out: list[str] = []

        for idx in ids:
            if idx == self.blank_id:
                # Word boundary for WordsMapper.
                out.append(" ")
                continue

            if idx in self.special_id_to_token:
                out.append(self.special_id_to_token[idx])
                continue

            if 0 <= idx < len(symbols):
                sym = symbols[idx]

                # FastSpeech2 ARPAbet symbols are usually stored as "@P", "@IY1", ...
                if sym.startswith("@"):
                    sym = sym[1:]

                out.append(sym)
                continue

            out.append(self.UNK)

        return out

    @cached_property
    @override
    def vocab_labels(self) -> list[str]:
        labels = list(symbols)
        labels.append(self.BLANK)
        labels.append(self.UNK)
        return labels

    def _encode_phone_chunk(
        self,
        phones: list[str],
        source_text: str | None = None,
    ) -> list[int]:
        """
        Encode a contiguous chunk of normal ARPAbet phones using FastSpeech2 text_to_sequence.

        If a phone is not in the FastSpeech2 symbol table, emit <UNK> and print
        the source text for debugging.
        """
        if len(phones) == 0:
            return []

        sequence: list[int] = []
        known_phone_chunk: list[str] = []
        unknown_phones: list[str] = []

        def flush_known_phone_chunk() -> None:
            nonlocal known_phone_chunk, sequence

            if len(known_phone_chunk) == 0:
                return

            phone_string = "{" + "}{".join(known_phone_chunk) + "}"

            # Keep the same behavior as the original FastSpeech2 preprocessing.
            phone_string = re.sub(r"\{[^\w\s]?\}", "{sp}", phone_string)
            phone_string = phone_string.replace("}{", " ")

            sequence.extend(
                text_to_sequence(
                    phone_string,
                    ["english_cleaners"],
                )
            )

            known_phone_chunk = []

        for phone in phones:
            # Special tokens should not normally enter this function,
            # but handle them defensively.
            if phone in self.special_token_to_id:
                flush_known_phone_chunk()
                sequence.append(self.special_token_to_id[phone])
                continue

            if self._is_known_phone(phone):
                known_phone_chunk.append(phone)
            else:
                flush_known_phone_chunk()
                sequence.append(self.unk_id)
                unknown_phones.append(phone)

        flush_known_phone_chunk()

        if len(unknown_phones) > 0:
            print(
                "[FastSpeech2Tokenizer] <UNK> emitted "
                + f"for unknown_phones={unknown_phones}, "
                + f"text={source_text!r}"
            )

        return sequence

    @staticmethod
    def _read_lexicon(lex_path: str | Path) -> dict[str, list[str]]:
        lexicon: dict[str, list[str]] = {}

        with open(lex_path, encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"\s+", line.strip())

                if len(parts) < 2:
                    continue

                word = parts[0]
                phones = parts[1:]

                key = word.lower()
                if key not in lexicon:
                    lexicon[key] = phones

        return lexicon

    def _normalize_text(self, text: str) -> str:
        text = text.rstrip(punctuation)
        text = english_cleaners(text)
        return text

    def _split_words(self, text: str) -> list[str]:
        """
        Split cleaned text into word units.

        Punctuation is removed from word edges.
        <BLANK> is inserted only between these word units.
        """
        text = self._normalize_text(text)

        words: list[str] = []
        for w in re.split(r"\s+", text):
            w = w.strip(punctuation)
            if w:
                words.append(w)

        return words

    def _word_to_phones(self, word: str) -> list[str]:
        key = word.lower()

        if key in self.lexicon:
            return self.lexicon[key]

        phones = self.g2p(word)

        # Remove spaces emitted by g2p_en.
        phones = [p for p in phones if p != " "]

        return phones

    def _phone_to_symbol(self, phone: str) -> str:
        """
        Convert an ARPAbet phone token to the FastSpeech2 symbol-table form.

        FastSpeech2 usually stores ARPAbet phones as '@AA', '@T', '@EH1', etc.
        Lexicon/G2P usually returns 'AA', 'T', 'EH1', etc.
        """
        if phone.startswith("@"):
            return phone

        return "@" + phone

    def _is_known_phone(self, phone: str) -> bool:
        """
        Check whether a normal ARPAbet phone exists in the FastSpeech2 symbol table.
        """
        symbol = self._phone_to_symbol(phone)
        return symbol in self.symbol_set

    @property
    @override
    def n_vocab(self) -> int:
        return len(symbols) + len(self.special_token_to_id)

    @property
    @override
    def ignore_symbols(self):
        return self._ignore_symbols

    @property
    @override
    def seperator_id(self):
        return self.blank_id


if __name__ == "__main__":
    tokenizer = ARPATokenizer()

    texts = [
        "Please call Stella.",
        "Hello world.",
        "This is a tokenizer test.",
        "I have two apples.",
        "FastSpeech two uses ARPAbet phones.",
    ]

    for idx, text in enumerate(texts):
        tokens = tokenizer(text)
        phones = tokenizer.to_token_string(text)

        print("=" * 80)
        print(f"[{idx}] text:", text)
        print("phones type:", type(phones))
        print("phones:", phones)
        print("tokens shape:", tokens.shape)
        print("tokens:", tokens)
        print("decoded:", tokenizer.decode(tokens.squeeze(0)))
