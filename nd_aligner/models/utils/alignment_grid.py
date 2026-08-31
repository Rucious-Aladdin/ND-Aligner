"""
Construction of phone and word interval grids from a token-level alignment.

An aligner produces one duration per text token. Turning that into intervals a
human or Praat can read requires deciding which tokens deserve an interval of
their own, and that decision depends on the tokenizer: character-level IPA emits
stress marks and closure tokens that occupy frames without naming a phone, and
punctuation occupies frames that are silence rather than speech.

`GridSymbolPolicy` collects those decisions in one place; the two build functions
apply them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nd_aligner.models.utils.lev_words_mapper import LevensteinWordsMapper
from nd_aligner.tokenizer.base_tokenizer import BaseTokenizer

Interval = tuple[float, float, str]

_WORD_STRIP_CHARS = ' \t\n\r;:,.!?¡¿—…"«»“”()[]{}'

# eSpeak character-level IPA emits stress marks as separate tokens.
_ESPEAK_STRESS_MARKS = frozenset({"ˈ", "ˌ"})

# Punctuation tokens occupy frames but do not name a phone.
_ESPEAK_PUNCTUATION = frozenset(",.;:!?—…")


def normalize_word(word: str) -> str:
    """Strip surrounding punctuation while keeping internal apostrophes."""
    return word.strip().strip(_WORD_STRIP_CHARS)


@dataclass(frozen=True)
class GridSymbolPolicy:
    """
    Which decoded symbols get an interval of their own, and what becomes of the
    frames of those that do not.

    Attributes:
        merge_into_next:
            Symbols whose frames are handed to the following symbol rather than
            forming an interval. These precede the thing they belong to: a
            stress mark precedes its vowel and a closure token precedes its
            stop, so in both cases the interval should begin where the marker
            does.
        silent:
            Symbols that keep their frames but are relabelled as silence, since
            they occupy time without naming a phone.
    """

    merge_into_next: frozenset[str] = frozenset()
    silent: frozenset[str] = frozenset()

    @classmethod
    def for_tokenizer(
        cls,
        tokenizer: BaseTokenizer,
        tokenizer_type: str,
    ) -> GridSymbolPolicy:
        """
        Derive a policy from the tokenizer that produced the symbols.

        Symbols the tokenizer reports as span-only occupy a token position
        without contributing a character, which is exactly the condition for
        folding them into the following interval; closure tokens reach the
        policy this way.
        """
        merge = frozenset(tokenizer.span_only_symbols)
        silent = frozenset()

        if tokenizer_type == "espeak":
            merge = merge | _ESPEAK_STRESS_MARKS
            silent = _ESPEAK_PUNCTUATION

        return cls(merge_into_next=merge, silent=silent)


def build_phone_grid(
    symbols: Sequence[str],
    token_ids: Sequence[int],
    durations: Sequence[int],
    sec_per_frame: float,
    separator_id: int,
    include_separator: bool,
    policy: GridSymbolPolicy,
) -> list[Interval]:
    """
    Build one interval per phone-bearing token.

    Args:
        symbols:
            Decoded symbol per token, aligned with `token_ids`.
        token_ids:
            The token ids the aligner consumed, truncated to the valid length.
        durations:
            Frames assigned to each token, same length as `token_ids`.
        sec_per_frame:
            Frame hop in seconds.
        separator_id:
            Token id of the word separator.
        include_separator:
            Emit an interval for the separator instead of dropping it.
        policy:
            Which symbols are folded away and which become silence.

    Returns:
        Intervals in seconds, in order, non-overlapping.
    """
    grid: list[Interval] = []

    current_frame = 0
    pending_start: int | None = None

    for symbol, token_id, duration in zip(symbols, token_ids, durations, strict=True):
        start_frame = current_frame
        current_frame += int(duration)

        if symbol in policy.merge_into_next:
            if pending_start is None:
                pending_start = start_frame
            continue

        if duration <= 0:
            continue

        if not include_separator and token_id == separator_id:
            pending_start = None
            continue

        if pending_start is not None:
            start_frame = pending_start
            pending_start = None

        label = "" if symbol in policy.silent else symbol

        grid.append(
            (
                start_frame * sec_per_frame,
                current_frame * sec_per_frame,
                label,
            )
        )

    return grid


def build_word_grid(
    ref_words: Sequence[str],
    symbols: Sequence[str],
    token_ids: Sequence[int],
    durations: Sequence[int],
    sec_per_frame: float,
    separator_id: int,
    include_separator: bool,
    mapper: LevensteinWordsMapper,
) -> list[Interval]:
    """
    Build one interval per reference word.

    The tokenizer need not spell a word the way the reference does, so the two
    sequences are matched by `mapper` and each matched span of tokens supplies
    the timing for the corresponding span of reference words.

    Args:
        ref_words:
            Reference words, already normalized.
        symbols:
            Decoded symbol per token.
        token_ids:
            The token ids the aligner consumed, truncated to the valid length.
        durations:
            Frames assigned to each token.
        sec_per_frame:
            Frame hop in seconds.
        separator_id:
            Token id of the word separator.
        include_separator:
            Emit an interval for separator tokens that fall between words.
        mapper:
            Matches reference words against decoded symbols.

    Returns:
        Intervals in seconds, in order, non-overlapping.
    """
    matched = mapper(ref_seqs=list(ref_words), hyp_seqs=list(symbols))

    cumulative: list[int] = [0]
    for duration in durations:
        cumulative.append(cumulative[-1] + int(duration))

    match_by_hyp_start: dict[int, tuple[slice, slice]] = {
        int(hyp_slice.start): (ref_slice, hyp_slice)
        for ref_slice, hyp_slice in zip(
            matched.ref_matched_indices,
            matched.hyp_matched_indices,
            strict=True,
        )
    }

    grid: list[Interval] = []
    token_idx = 0
    text_length = len(token_ids)

    while token_idx < text_length:
        matched_item = match_by_hyp_start.get(token_idx)

        if matched_item is not None:
            ref_slice, hyp_slice = matched_item

            start_frame = cumulative[int(hyp_slice.start)]
            end_frame = cumulative[int(hyp_slice.stop)]

            if end_frame > start_frame:
                label = " ".join(ref_words[int(ref_slice.start) : int(ref_slice.stop)])
                grid.append(
                    (
                        start_frame * sec_per_frame,
                        end_frame * sec_per_frame,
                        label,
                    )
                )

            token_idx = int(hyp_slice.stop)
            continue

        if include_separator and token_ids[token_idx] == separator_id:
            start_frame = cumulative[token_idx]
            end_frame = cumulative[token_idx + 1]

            if end_frame > start_frame:
                grid.append(
                    (
                        start_frame * sec_per_frame,
                        end_frame * sec_per_frame,
                        " ",
                    )
                )

        token_idx += 1

    return grid


def to_praat_intervals(
    grid: Sequence[Interval],
    duration: float,
    silence_label: str = "",
) -> list[Interval]:
    """
    Pad a grid into a contiguous interval tier covering ``[0, duration]``.

    Praat expects a tier to have no gaps, so the space between predicted
    intervals is filled with `silence_label` and consecutive silences are
    collapsed. Interval bounds are clipped, which absorbs both the rounding of
    the final frame and any overlap left by the merging in
    `build_phone_grid`.
    """
    intervals: list[Interval] = []
    cursor = 0.0

    for start, end, label in grid:
        start = max(start, cursor)
        end = min(end, duration)

        if end <= start:
            continue

        if start > cursor:
            intervals.append((cursor, start, silence_label))

        intervals.append((start, end, label if label.strip() else silence_label))
        cursor = end

    if cursor < duration:
        intervals.append((cursor, duration, silence_label))

    merged: list[Interval] = []
    for start, end, label in intervals:
        if merged and label == silence_label and merged[-1][2] == silence_label:
            merged[-1] = (merged[-1][0], end, silence_label)
        else:
            merged.append((start, end, label))

    return merged
