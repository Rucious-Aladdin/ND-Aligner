from pathlib import Path
from typing import NamedTuple

from tts.tokenizer.base_tokenizer import BaseTokenizer


class MatchedWords(NamedTuple):
    matched_sequences: list[str]

    # ref: word span over original TIMIT .wrd words, half-open [start, stop)
    # hyp: token span over original model input token sequence, half-open [start, stop)
    ref_matched_indices: list[slice]
    hyp_matched_indices: list[slice]

    ref_seqs: list[str]
    hyp_seqs: list[str]

    # covered reference words / total reference words
    coverage_ratio: float


class WordUnit(NamedTuple):
    key: str
    span: slice


class TimitWordSegment(NamedTuple):
    start_sample: int
    end_sample: int
    word: str

    @property
    def start_sec(self) -> float:
        return self.start_sample / 16000.0

    @property
    def end_sec(self) -> float:
        return self.end_sample / 16000.0


def normalize_ref_word(word: str) -> str:
    # Keep internal apostrophe: don't, we're, Jennifer's, etc.
    return word.strip().lower().strip(' \t\n\r;:,.!?¡¿—…"«»“”()[]{}')


def normalize_match_key(word: str) -> str:
    return word.strip().lower().replace("ˈ", "").replace("ˌ", "").replace("ɚɹ", "ɚ")


def split_ipa_words(ipa_text: str) -> list[str]:
    punctuation = set(';:,.!?¡¿—…"«»“”')

    words: list[str] = []

    for chunk in ipa_text.split(" "):
        chunk = chunk.strip()
        if not chunk:
            continue

        if chunk in punctuation:
            continue

        words.append(normalize_match_key(chunk))

    return words


class WordsMapper:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        hyp_ignore_symbols: set[str] | None = None,
        max_ref_words_per_hyp_word: int = 5,
    ) -> None:
        self.tokenizer = tokenizer
        self.hyp_ignore_symbols = set(hyp_ignore_symbols or set())
        self.max_ref_words_per_hyp_word = max_ref_words_per_hyp_word

    def __call__(
        self,
        ref_seqs: list[str],
        hyp_seqs: list[str],
    ) -> MatchedWords:
        hyp_units = self._make_hyp_units(hyp_seqs)
        ref_units = self._make_ref_units_from_hyp_units(
            ref_words=ref_seqs,
            hyp_units=hyp_units,
        )

        ref_keys = [unit.key for unit in ref_units]
        hyp_keys = [unit.key for unit in hyp_units]

        ref_match_ids, hyp_match_ids = self._lcs_match(ref_keys, hyp_keys)

        matched_sequences = [ref_units[i].key for i in ref_match_ids]
        ref_matched_indices = [ref_units[i].span for i in ref_match_ids]
        hyp_matched_indices = [hyp_units[i].span for i in hyp_match_ids]

        num_covered_ref_words = sum(
            ref_span.stop - ref_span.start for ref_span in ref_matched_indices
        )

        num_ref_words = len([word for word in ref_seqs if normalize_ref_word(word)])

        coverage_ratio = num_covered_ref_words / num_ref_words if num_ref_words > 0 else 0.0

        return MatchedWords(
            matched_sequences=matched_sequences,
            ref_matched_indices=ref_matched_indices,
            hyp_matched_indices=hyp_matched_indices,
            ref_seqs=ref_seqs,
            hyp_seqs=hyp_seqs,
            coverage_ratio=coverage_ratio,
        )

    def _make_hyp_units(self, hyp_symbols: list[str]) -> list[WordUnit]:
        units: list[WordUnit] = []

        chars: list[str] = []
        start: int | None = None
        stop: int | None = None

        def flush() -> None:
            nonlocal chars, start, stop

            if start is None or stop is None:
                chars = []
                start = None
                stop = None
                return

            key = normalize_match_key("".join(chars))
            if key:
                units.append(
                    WordUnit(
                        key=key,
                        span=slice(start, stop),
                    )
                )

            chars = []
            start = None
            stop = None

        for idx, symbol in enumerate(hyp_symbols):
            if symbol == " ":
                flush()
                continue

            if symbol in self.hyp_ignore_symbols:
                continue

            if start is None:
                start = idx

            chars.append(symbol)
            stop = idx + 1

        flush()

        return units

    @staticmethod
    def _lev_dist(s1: str, s2: str) -> int:
        """Calculate the Levenshtein distance between two strings."""
        n1, n2 = len(s1), len(s2)
        row = list(range(n2 + 1))
        for i in range(1, n1 + 1):
            new_row = [i]
            for j in range(1, n2 + 1):
                if s1[i - 1] == s2[j - 1]:
                    new_row.append(row[j - 1])
                else:
                    new_row.append(1 + min(row[j], new_row[j - 1], row[j - 1]))
            row = new_row
        return row[-1]

    def _make_ref_units_from_hyp_units(
        self,
        ref_words: list[str],
        hyp_units: list[WordUnit],
    ) -> list[WordUnit]:
        ref_norm = [normalize_ref_word(word) for word in ref_words]

        n = len(ref_norm)
        m = len(hyp_units)

        span_key_cache: dict[tuple[int, int], str | None] = {}

        def span_to_key(start: int, stop: int) -> str | None:
            cache_key = (start, stop)
            if cache_key in span_key_cache:
                return span_key_cache[cache_key]

            words = [word for word in ref_norm[start:stop] if word]
            if not words:
                span_key_cache[cache_key] = None
                return None

            text = " ".join(words)
            ipa_text = self.tokenizer.to_token_string(text)
            chunks = split_ipa_words(ipa_text)

            key = "".join(normalize_match_key(c) for c in chunks)
            span_key_cache[cache_key] = key
            return key

        dp = [[float("inf")] * (n + 1) for _ in range(m + 1)]
        prev = [[None] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0.0

        for k, hyp_unit in enumerate(hyp_units):
            hyp_key = hyp_unit.key

            for start in range(n + 1):
                if dp[k][start] == float("inf"):
                    continue

                max_stop = min(n, start + self.max_ref_words_per_hyp_word)

                for stop in range(start, max_stop + 1):
                    if stop == start:
                        ref_key = ""
                        dist = len(hyp_key)
                        cost = dp[k][start] + dist + 2.0  # 0개 맵핑은 페널티를 주어 남발 방지
                    else:
                        ref_key = span_to_key(start, stop)
                        if ref_key is None:
                            continue

                        dist = self._lev_dist(ref_key, hyp_key)
                        cost = dp[k][start] + dist + 0.1 * (stop - start - 1)

                    if cost < dp[k + 1][stop]:
                        dp[k + 1][stop] = cost
                        prev[k + 1][stop] = start  # type: ignore

        if prev[m][n] is None:
            hyp_keys = [unit.key for unit in hyp_units]
            raise ValueError(
                "Failed to align TIMIT words to tokenizer IPA chunks.\n"
                + f"ref_words={ref_norm}\n"
                + f"hyp_words={hyp_keys}\n"
            )

        # Backtrace
        spans_rev: list[slice] = []
        i = n

        for k in range(m, 0, -1):
            start = prev[k][i]
            if start is None:
                raise RuntimeError("Invalid DP backtrace.")

            spans_rev.append(slice(start, i))
            i = start

        spans_rev.reverse()

        return [
            WordUnit(
                key=hyp_unit.key,
                span=ref_span,
            )
            for hyp_unit, ref_span in zip(hyp_units, spans_rev, strict=True)
        ]

    @staticmethod
    def _lcs_match(
        ref: list[str],
        hyp: list[str],
    ) -> tuple[list[int], list[int]]:
        n = len(ref)
        m = len(hyp)

        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n):
            for j in range(m):
                if ref[i] == hyp[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(
                        dp[i][j + 1],
                        dp[i + 1][j],
                    )

        ref_ids: list[int] = []
        hyp_ids: list[int] = []

        i = n
        j = m

        while i > 0 and j > 0:
            if ref[i - 1] == hyp[j - 1]:
                ref_ids.append(i - 1)
                hyp_ids.append(j - 1)
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1

        ref_ids.reverse()
        hyp_ids.reverse()

        return ref_ids, hyp_ids


if __name__ == "__main__":
    import random

    from tqdm import tqdm

    from tts.tokenizer.arpa_tokenizer import ARPATokenizer
    from tts.tokenizer.espeak_tokenizer import ESPEAKTokenizer

    def read_timit_wrd(wrd_path: str | Path) -> list[TimitWordSegment]:
        segments: list[TimitWordSegment] = []

        with open(wrd_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=2)
                if len(parts) != 3:
                    raise ValueError(f"Invalid WRD line in {wrd_path}: {line!r}")

                start, end, word = parts
                segments.append(
                    TimitWordSegment(
                        start_sample=int(start),
                        end_sample=int(end),
                        word=word,
                    )
                )

        return segments

    def read_timit_txt(txt_path: str | Path) -> str:
        with open(txt_path, "r", encoding="utf-8") as f:
            line = f.read().strip()

        parts = line.split(maxsplit=2)

        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            return parts[2]

        return line

    def find_matching_txt_path(wrd_path: Path) -> Path:
        for suffix in (".TXT", ".txt"):
            txt_path = wrd_path.with_suffix(suffix)
            if txt_path.exists():
                return txt_path

        raise FileNotFoundError(f"No matching TXT file found for {wrd_path}")

    def get_ref_end_sec(
        wrd_segments: list[TimitWordSegment],
        ref_index: int | slice,
    ) -> float:
        if isinstance(ref_index, int):
            return wrd_segments[ref_index].end_sec

        if ref_index.stop is None:
            raise ValueError(f"Invalid ref slice without stop: {ref_index}")

        return wrd_segments[ref_index.stop - 1].end_sec

    def get_ref_words(
        ref_words: list[str],
        ref_index: int | slice,
    ) -> list[str]:
        if isinstance(ref_index, int):
            return [ref_words[ref_index]]

        return ref_words[ref_index]

    def format_slices(slices: list[slice]) -> list[tuple[int | None, int | None]]:
        return [(s.start, s.stop) for s in slices]

    def run_one_sample(
        wrd_path: Path,
        *,
        tokenizer: BaseTokenizer,
        mapper: WordsMapper,
        verbose: bool,
    ) -> float:
        txt_path = find_matching_txt_path(wrd_path)

        wrd_segments = read_timit_wrd(wrd_path)
        ref_words = [seg.word for seg in wrd_segments]

        text = read_timit_txt(txt_path)

        # Same tokenization path as TTSDataset:
        token_ids = tokenizer(text).squeeze(0)

        decoded = tokenizer.decode(token_ids)
        assert isinstance(decoded, str)

        hyp_symbols_raw = tokenizer.decode_to_symbols(token_ids)

        if len(hyp_symbols_raw) > 0 and isinstance(hyp_symbols_raw[0], list):
            raise ValueError("Expected 1D token symbols.")

        hyp_symbols = [str(sym) for sym in hyp_symbols_raw]

        assert len(hyp_symbols) == token_ids.numel(), (
            f"len(hyp_symbols)={len(hyp_symbols)}, token_ids.numel()={token_ids.numel()}\n"
            f"decoded={decoded!r}\n"
            f"hyp_symbols={hyp_symbols}"
        )

        matched = mapper(
            ref_seqs=ref_words,
            hyp_seqs=hyp_symbols,
        )

        if verbose:
            ref_end_sec = [
                get_ref_end_sec(wrd_segments, ref_index)
                for ref_index in matched.ref_matched_indices
            ]

            hyp_end_token_indices = [
                hyp_slice.stop - 1 for hyp_slice in matched.hyp_matched_indices
            ]

            ref_word_groups = [
                get_ref_words(ref_words, ref_index) for ref_index in matched.ref_matched_indices
            ]

            hyp_word_strings = [
                "".join(hyp_symbols[hyp_slice]) for hyp_slice in matched.hyp_matched_indices
            ]

            print("=" * 80)
            print(f"WRD: {wrd_path}")
            print(f"TXT: {txt_path}")
            print(f"Text:    {text}")
            print(f"IPA:     {tokenizer.to_token_string(text)}")
            print(f"Decoded: {decoded}")
            print()

            print("Ref words:")
            print(ref_words)
            print()

            print("Matched normalized chunks:")
            print(matched.matched_sequences)
            print()

            print("Matched ref word groups:")
            print(ref_word_groups)
            print()

            print("Matched hyp token strings:")
            print(hyp_word_strings)
            print()

            print("Ref matched indices:")
            print(matched.ref_matched_indices)
            print()

            print("Hyp matched slices:")
            print(format_slices(matched.hyp_matched_indices))
            print()

            print("Ref end sec:")
            print(ref_end_sec)
            print()

            print("Hyp end token indices:")
            print(hyp_end_token_indices)
            print()

            print(f"Coverage: {matched.coverage_ratio:.4f}")

        return matched.coverage_ratio

    # tokenizer = ESPEAKTokenizer()
    tokenizer = ARPATokenizer()

    mapper = WordsMapper(
        tokenizer=tokenizer,
        hyp_ignore_symbols=tokenizer.ignore_symbols,
        max_ref_words_per_hyp_word=5,
    )

    # -------------------------------------------------------------------------
    # Full TIMIT TEST coverage check
    # -------------------------------------------------------------------------
    timit_test_root = Path("/shared/data_zfs/blue2959/TIMIT/TEST")

    wrd_paths = sorted(timit_test_root.rglob("*.WRD"))
    wrd_paths += sorted(timit_test_root.rglob("*.wrd"))

    if not wrd_paths:
        raise FileNotFoundError(f"No WRD files found under {timit_test_root}")

    coverage_ratios: list[float] = []
    num_warnings = 0
    num_failed = 0

    merged_2plus_examples: list[tuple[Path, list[tuple[list[str], str]]]] = []
    merged_3plus_examples: list[tuple[Path, list[tuple[list[str], str]]]] = []

    tqdm.write("=" * 80)
    tqdm.write(f"[Info] Running word-mapping coverage check on {len(wrd_paths)} TIMIT TEST samples")
    tqdm.write(f"[Info] Root: {timit_test_root}")

    for wrd_path in tqdm(wrd_paths, desc="Checking TIMIT TEST word coverage"):
        try:
            txt_path = find_matching_txt_path(wrd_path)

            wrd_segments = read_timit_wrd(wrd_path)
            ref_words = [seg.word for seg in wrd_segments]

            text = read_timit_txt(txt_path)
            token_ids = tokenizer(text).squeeze(0)

            decoded = tokenizer.decode(token_ids)
            assert isinstance(decoded, str)

            hyp_symbols = list(decoded)

            matched = mapper(
                ref_seqs=ref_words,
                hyp_seqs=hyp_symbols,
            )

            coverage_ratios.append(matched.coverage_ratio)
            if matched.coverage_ratio != 1.0:
                num_warnings += 1

            # 🌟 전체 시퀀스 쌍 수집
            sequence_pairs = []
            has_2plus = False
            has_3plus = False

            for ref_span, hyp_span in zip(matched.ref_matched_indices, matched.hyp_matched_indices):
                num_words = ref_span.stop - ref_span.start
                refs = ref_words[ref_span.start : ref_span.stop]
                hyp = "".join(hyp_symbols[hyp_span])

                sequence_pairs.append((refs, hyp))

                if num_words >= 2:
                    has_2plus = True
                if num_words >= 3:
                    has_3plus = True

            if has_2plus:
                merged_2plus_examples.append((wrd_path, sequence_pairs))
            if has_3plus:
                merged_3plus_examples.append((wrd_path, sequence_pairs))

        except Exception as e:
            num_failed += 1
            tqdm.write("\n" + "=" * 80)
            tqdm.write(f"[Warning] Failed to process sample: {wrd_path}")
            tqdm.write(f"Reason: {repr(e)}")

    mean_coverage = sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0.0

    tqdm.write("=" * 80)
    tqdm.write("[Result] TIMIT TEST word-mapping coverage")
    tqdm.write(f"Num processed: {len(coverage_ratios)} / {len(wrd_paths)}")
    tqdm.write(f"Num coverage warnings: {num_warnings}")
    tqdm.write(f"Num failed: {num_failed}")
    tqdm.write(f"Mean coverage ratio: {mean_coverage:.6f}")

    # 🌟 랜덤 출력 헬퍼 함수
    def print_examples(examples, title, max_samples=5):
        if not examples:
            tqdm.write(f"\n[Info] No examples found for: {title}")
            return

        tqdm.write("\n" + "=" * 80)
        tqdm.write(f"[Info] 🎲 {title} (Showing up to {max_samples} random samples)")
        tqdm.write("=" * 80)

        sampled = random.sample(examples, min(max_samples, len(examples)))
        for wrd_path, seq_pairs in sampled:
            tqdm.write(f"\n📁 {wrd_path.name}")
            for refs, hyp in seq_pairs:
                if len(refs) >= 3:
                    tqdm.write(f"  🔥 {str(refs):<25} -->  '{hyp}'")
                elif len(refs) == 2:
                    tqdm.write(f"  ✨ {str(refs):<25} -->  '{hyp}'")
                else:
                    tqdm.write(f"     {str(refs):<25} -->  '{hyp}'")
            tqdm.write("-" * 50)

    # 2개짜리와 3개짜리가 병합된 시퀀스를 각각 최대 5개씩 뽑아서 출력합니다.
    print_examples(merged_2plus_examples, "Sequences with 2+ merged words", max_samples=5)
    print_examples(merged_3plus_examples, "Sequences with 3+ merged words", max_samples=5)

    wrd_paths = sorted(timit_test_root.rglob("*.WRD"))
    wrd_paths += sorted(timit_test_root.rglob("*.wrd"))

    if not wrd_paths:
        raise FileNotFoundError(f"No WRD files found under {timit_test_root}")

    # -------------------------------------------------------------------------
    # Verbose examples using run_one_sample
    # -------------------------------------------------------------------------
    example_wrd_paths = random.sample(
        wrd_paths,
        k=min(2, len(wrd_paths)),
    )

    tqdm.write("=" * 80)
    tqdm.write("[Info] Running 2 verbose run_one_sample examples")
    tqdm.write("=" * 80)

    for example_idx, example_wrd_path in enumerate(example_wrd_paths):
        tqdm.write(f"\n[Example {example_idx}] {example_wrd_path}")

        try:
            coverage = run_one_sample(
                example_wrd_path,
                tokenizer=tokenizer,
                mapper=mapper,
                verbose=True,
            )
            tqdm.write(f"[Example {example_idx}] Coverage: {coverage:.4f}")

        except Exception as e:
            tqdm.write(f"[Example {example_idx}] Failed: {repr(e)}")

    coverage_ratios: list[float] = []
    num_warnings = 0
    num_failed = 0
