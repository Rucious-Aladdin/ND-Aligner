from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple, override

import librosa
import noisereduce as nr
import torch
import torch.nn as nn
from silero_vad import get_speech_timestamps, load_silero_vad

from tts.audio.linear_spectrogram import LinearSpecExtractor
from tts.audio.mel_spectrogram import MelSpecExtractor
from tts.audio.utils.suppress_impulsive_peaks import suppress_impulsive_peaks
from tts.config.ndaligner.data_config import AudioConfigs
from tts.config.preprocess.preprocess_config import PreprocessConfigs
from tts.models.modules.spk_encoder import ECAPASpeakerEncoder, ResemblyzerSpeakerEncoder
from tts.models.utils.lev_words_mapper import LevensteinWordsMapper
from tts.tokenizer.load_tokenizer import load_tokenizer


class AlignerInput(NamedTuple):
    """Canonical ND-Aligner input tuple."""

    x: torch.Tensor
    x_lengths: torch.Tensor
    y: torch.Tensor
    y_lengths: torch.Tensor
    cond: torch.Tensor | None


class AlignerInputWithAudio(NamedTuple):
    """
    ND-Aligner input plus normalized 16-kHz reference waveform.

    TIMITBenchMarker uses wav_16k for optional reconstruction MCD-DTW.
    The first five fields intentionally match AlignerInput.
    """

    x: torch.Tensor
    x_lengths: torch.Tensor
    y: torch.Tensor
    y_lengths: torch.Tensor
    cond: torch.Tensor | None
    wav_16k: torch.Tensor
    wav_16k_lengths: torch.Tensor
    wav_16k_start_offset: torch.Tensor
    wav_16k_end_offset: torch.Tensor
    texts: list[str]

    def as_aligner_input(self) -> AlignerInput:
        return AlignerInput(
            x=self.x,
            x_lengths=self.x_lengths,
            y=self.y,
            y_lengths=self.y_lengths,
            cond=self.cond,
        )


class AlignerInputMaker(nn.Module):
    """
    Build ND-Aligner inputs from waveform paths and already-read scripts.

    Responsibilities:
      - tokenize text
      - load audio at model sampling rate and 16 kHz speaker/reference rate
      - optional peak normalization
      - extract mel or linear spectrogram
      - extract speaker condition vector
      - pad variable-length batches

    Returns:
        x:          (B, T_text)
        x_lengths:  (B,)
        y:          (B, C_spec, T_spec)
        y_lengths:  (B,)
        cond:       (B, C_cond) or None
    """

    def __init__(
        self,
        audio_config: AudioConfigs,
        preprocess_config: PreprocessConfigs,
        tokenizer_type: str,
        fastspeech2_lexicon_path: str = "",
        zero_nonspeech_region: bool = False,
        trim_nonspeech_region: bool = True,
        suppress_impulsive_peak: bool = False,
        reduce_noise: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.audio_config = audio_config
        self.preprocess_config = preprocess_config
        self.device = torch.device(device)

        self.feature_type = audio_config.feature_type
        self.model_sr = int(audio_config.sr)

        # Amplitude normalization.
        self.peak_normalize = bool(preprocess_config.peak_normalize)
        self.peak_target = float(preprocess_config.peak_target)

        self.tokenizer = load_tokenizer(
            tokenizer_type=tokenizer_type,
            fastspeech2_lexicon_path=fastspeech2_lexicon_path,
        )

        if audio_config.feature_type == "mel":
            self.spec_extractor = MelSpecExtractor(
                sr=audio_config.sr,
                n_mels=audio_config.n_mels,
                n_fft=audio_config.n_fft,
                hop_length=audio_config.hop_length,
                win_length=audio_config.win_length,
                fmin=audio_config.f_min,
                fmax=audio_config.f_max,
            ).to(device)
        elif audio_config.feature_type == "linspec":
            self.spec_extractor = LinearSpecExtractor(
                sr=audio_config.sr,
                n_fft=audio_config.n_fft,
                hop_length=audio_config.hop_length,
                win_length=audio_config.win_length,
                fmax=audio_config.f_max,
            ).to(device)
        else:
            raise ValueError(f"Unsupported feature_type: {audio_config.feature_type!r}")

        self.zero_nonspeech_region = bool(zero_nonspeech_region)
        self.trim_nonspeech_region = bool(trim_nonspeech_region)
        self.suppress_impulsive_peaks = bool(suppress_impulsive_peak)
        self.reduce_noise = bool(reduce_noise)

        use_silero_vad = self.zero_nonspeech_region or self.trim_nonspeech_region
        self.silero_model = load_silero_vad(onnx=True) if use_silero_vad else None  # FOR SPEED!

        if self.preprocess_config.spk_encoder_type == "ecapa-tdnn":
            self.speaker_encoder = ECAPASpeakerEncoder(device=str(self.device))
        elif self.preprocess_config.spk_encoder_type == "resemblyzer":
            self.speaker_encoder = ResemblyzerSpeakerEncoder(device=str(device))
        self.speaker_encoder.eval()

        self.word_mapper = LevensteinWordsMapper(
            tokenizer=self.tokenizer,
            hyp_ignore_symbols=self.tokenizer.ignore_symbols,
        )

    @torch.no_grad()
    @override
    def forward(
        self,
        wav_paths: list[str | Path],
        scripts: list[str],
    ) -> AlignerInput:
        batch = self.make_with_audio(
            wav_paths=wav_paths,
            scripts=scripts,
        )
        return batch.as_aligner_input()

    @torch.no_grad()
    def make_with_audio(
        self,
        wav_paths: list[str | Path],
        scripts: list[str],
    ) -> AlignerInputWithAudio:
        wav_path_list = [Path(path) for path in wav_paths]
        texts = list(scripts)

        if len(wav_path_list) != len(texts):
            raise ValueError(
                "wav_paths and scripts must have the same batch size: "
                + f"{len(wav_path_list)} != {len(texts)}"
            )

        if len(wav_path_list) == 0:
            raise ValueError("Empty input batch.")

        token_seqs = [self.tokenizer(text).squeeze(0).long() for text in texts]
        x, x_lengths = self._pad_token_sequences(token_seqs, device=self.device)

        wav_16k_list: list[torch.Tensor] = []
        wav_model_list: list[torch.Tensor] = []
        wav_16k_start_offset_list: list[int] = []
        wav_16k_end_offset_list: list[int] = []

        for wav_path in wav_path_list:
            (
                wav_16k_cpu,
                wav_model_cpu,
                wav_16k_start_offset,
                wav_16k_end_offset,
            ) = self._load_waveforms(wav_path)

            wav_16k_list.append(wav_16k_cpu.squeeze(0))
            wav_model_list.append(wav_model_cpu.squeeze(0))
            wav_16k_start_offset_list.append(wav_16k_start_offset)
            wav_16k_end_offset_list.append(wav_16k_end_offset)

        wav_model, wav_model_lengths = self._pad_waveforms(
            wav_model_list,
            device=self.device,
        )
        wav_16k, wav_16k_lengths = self._pad_waveforms(
            wav_16k_list,
            device=self.device,
        )

        y = self.spec_extractor(wav_model)
        y_lengths = self.compute_spec_lengths(wav_model_lengths)

        cond = self.speaker_encoder(wav_16k)
        cond = cond.view(len(wav_path_list), -1).to(device=self.device, dtype=y.dtype)

        wav_16k_start_offset = torch.tensor(
            wav_16k_start_offset_list,
            dtype=torch.long,
            device=self.device,
        )
        wav_16k_end_offset = torch.tensor(
            wav_16k_end_offset_list,
            dtype=torch.long,
            device=self.device,
        )
        return AlignerInputWithAudio(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            cond=cond,
            wav_16k=wav_16k,
            wav_16k_lengths=wav_16k_lengths,
            wav_16k_start_offset=wav_16k_start_offset,
            wav_16k_end_offset=wav_16k_end_offset,
            texts=texts,
        )

    def compute_spec_lengths(self, wav_lengths: torch.Tensor) -> torch.Tensor:
        n_fft = int(self.audio_config.n_fft)
        hop = int(self.audio_config.hop_length)
        pad = int((n_fft - hop) / 2)
        return (
            torch.div(
                wav_lengths + 2 * pad - n_fft,
                hop,
                rounding_mode="floor",
            )
            + 1
        )

    def _load_waveforms(
        self,
        wav_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """
        Load waveform at 16 kHz and model sampling rate.

        When trim_nonspeech_region is enabled, both the 16-kHz waveform
        and model-rate waveform are cropped from the beginning of the
        first detected speech region to the end of the last detected
        speech region.

        Offsets are represented in the original 16-kHz sample coordinate:

            start_offset: inclusive
            end_offset:   inclusive

        If trimming is disabled or speech is not detected:

            start_offset = 0
            end_offset = original_16k_length - 1
        """
        wav_16k_np, _ = librosa.load(
            str(wav_path),
            sr=16_000,
            mono=True,
        )

        if self.reduce_noise:
            wav_16k_np = nr.reduce_noise(
                y=wav_16k_np,
                sr=16_000,
                stationary=False,
                prop_decrease=0.8,
                n_fft=512,
            )

        wav_16k_cpu = torch.from_numpy(wav_16k_np).float().unsqueeze(0)

        if self.suppress_impulsive_peaks:
            wav_16k_cpu = suppress_impulsive_peaks(wav_16k_cpu)

        original_16k_length = int(wav_16k_cpu.size(-1))

        if original_16k_length == 0:
            raise RuntimeError(f"Loaded an empty waveform: {wav_path}")

        if self.model_sr == 16000:
            wav_model_cpu = wav_16k_cpu.clone()
        else:
            wav_model_np, _ = librosa.load(
                str(wav_path),
                sr=self.model_sr,
                mono=True,
            )
            wav_model_cpu = torch.from_numpy(wav_model_np).float().unsqueeze(0)

        if self.peak_normalize:
            wav_16k_cpu, wav_model_cpu = self._peak_normalize_pair(
                wav_16k=wav_16k_cpu,
                wav_model=wav_model_cpu,
                peak_target=self.peak_target,
            )

        # Default: the returned waveform covers the entire original waveform.
        wav_16k_start_offset = 0
        wav_16k_end_offset = original_16k_length - 1

        use_vad = self.zero_nonspeech_region or self.trim_nonspeech_region

        if use_vad:
            wav_for_vad = wav_16k_cpu.squeeze(0).float().contiguous()

            with torch.inference_mode():
                speech_regions = get_speech_timestamps(
                    wav_for_vad,
                    self.silero_model,
                    sampling_rate=16_000,
                    threshold=0.3,
                    neg_threshold=None,  # type: ignore
                    min_speech_duration_ms=10,
                    min_silence_duration_ms=10,
                    speech_pad_ms=30,
                    return_seconds=False,
                )

            if speech_regions:
                speech_start = int(speech_regions[0]["start"])
                speech_end_exclusive = int(speech_regions[-1]["end"])

                speech_start = max(
                    0,
                    min(speech_start, original_16k_length - 1),
                )
                speech_end_exclusive = max(
                    speech_start + 1,
                    min(speech_end_exclusive, original_16k_length),
                )

                if self.zero_nonspeech_region:
                    wav_16k_cpu = wav_16k_cpu.clone()
                    wav_16k_cpu[:, :speech_start] = 0.0
                    wav_16k_cpu[:, speech_end_exclusive:] = 0.0

                if self.trim_nonspeech_region:
                    wav_16k_start_offset = speech_start
                    wav_16k_end_offset = speech_end_exclusive - 1

                    wav_16k_cpu = wav_16k_cpu[
                        :,
                        speech_start:speech_end_exclusive,
                    ]

                    if self.model_sr == 16000:
                        model_start = speech_start
                        model_end_exclusive = speech_end_exclusive
                    else:
                        model_start = (speech_start * self.model_sr) // 16000

                        model_end_exclusive = (
                            speech_end_exclusive * self.model_sr + 16000 - 1
                        ) // 16000

                        model_length = int(wav_model_cpu.size(-1))

                        model_start = max(
                            0,
                            min(model_start, model_length - 1),
                        )
                        model_end_exclusive = max(
                            model_start + 1,
                            min(model_end_exclusive, model_length),
                        )

                    wav_model_cpu = wav_model_cpu[
                        :,
                        model_start:model_end_exclusive,
                    ]

        return (
            wav_16k_cpu,
            wav_model_cpu,
            wav_16k_start_offset,
            wav_16k_end_offset,
        )

    @staticmethod
    def _peak_normalize_pair(
        wav_16k: torch.Tensor,
        wav_model: torch.Tensor,
        peak_target: float = 0.6,
        eps: float = 1e-7,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        peak = wav_16k.abs().max()

        if peak < eps:
            return wav_16k, wav_model

        gain = float(peak_target / peak.item())

        wav_16k = torch.clamp(wav_16k * gain, -1.0, 1.0)
        wav_model = torch.clamp(wav_model * gain, -1.0, 1.0)

        return wav_16k, wav_model

    def _pad_token_sequences(
        self,
        seqs: list[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor(
            [seq.numel() for seq in seqs],
            dtype=torch.long,
            device=device,
        )
        max_len = int(lengths.max().item())
        pad_id = int(getattr(self.tokenizer, "pad_id", 0))

        x = torch.full(
            (len(seqs), max_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )

        for idx, seq in enumerate(seqs):
            seq = seq.to(device=device, dtype=torch.long)
            x[idx, : seq.numel()] = seq

        return x, lengths

    @staticmethod
    def _pad_waveforms(
        wavs: list[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor(
            [wav.numel() for wav in wavs],
            dtype=torch.long,
            device=device,
        )
        max_len = int(lengths.max().item())

        padded = torch.zeros(
            len(wavs),
            max_len,
            dtype=torch.float32,
            device=device,
        )

        for idx, wav in enumerate(wavs):
            wav = wav.to(device=device, dtype=torch.float32)
            padded[idx, : wav.numel()] = wav

        return padded, lengths
