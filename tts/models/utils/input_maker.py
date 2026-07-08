from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, override

import librosa
import torch
import torch.nn as nn

from tts.audio.linear_spectrogram import LinearSpecExtractor
from tts.audio.mel_spectrogram import MelSpecExtractor
from tts.config.ndaligner.data_config import AudioConfigs
from tts.config.preprocess.preprocess_config import PreprocessConfigs
from tts.models.modules.spk_encoder import ECAPASpeakerEncoder
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
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.audio_config = audio_config
        self.preprocess_config = preprocess_config
        self.device = torch.device(device)

        self.feature_type = audio_config.feature_type
        self.model_sr = int(audio_config.sr)

        # Amplitude normalization.
        # RMS normalization has been removed from PreprocessConfigs.
        self.peak_normalize = bool(preprocess_config.peak_normalize)
        self.peak_target = float(preprocess_config.peak_target)

        if self.peak_target <= 0.0:
            raise ValueError(f"peak_target must be positive, got {self.peak_target}")

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

        self.speaker_encoder = ECAPASpeakerEncoder(device=str(self.device))
        self.speaker_encoder.eval()

    @torch.no_grad()
    @override
    def forward(
        self,
        wav_paths: str | Path | list[str | Path],
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
        wav_paths: str | Path | list[str | Path],
        scripts: list[str],
    ) -> AlignerInputWithAudio:
        wav_path_list = self._as_path_list(wav_paths)
        texts = self._as_script_list(scripts)

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

        for wav_path in wav_path_list:
            wav_16k_cpu, wav_model_cpu = self._load_waveforms(wav_path)

            wav_16k_list.append(wav_16k_cpu.squeeze(0))
            wav_model_list.append(wav_model_cpu.squeeze(0))

        wav_model, wav_model_lengths = self._pad_waveforms(
            wav_model_list,
            device=self.device,
        )
        wav_16k, wav_16k_lengths = self._pad_waveforms(
            wav_16k_list,
            device=self.device,
        )

        y = self.spec_extractor(wav_model)
        y_lengths = self._compute_spec_lengths(wav_model_lengths)

        self._validate_spec_lengths(
            y=y,
            y_lengths=y_lengths,
            wav_lengths=wav_model_lengths,
        )

        cond = self._make_condition(
            wav_16k=wav_16k,
            y_dtype=y.dtype,
            batch_size=len(wav_path_list),
        )

        return AlignerInputWithAudio(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            cond=cond,
            wav_16k=wav_16k,
            wav_16k_lengths=wav_16k_lengths,
            texts=texts,
        )

    def _load_waveforms(self, wav_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load waveform at 16 kHz and model sampling rate.

        If peak normalization is enabled, the gain is computed from the 16-kHz
        waveform and applied to both waveforms. This keeps speaker embedding,
        reference waveform, and model feature extraction amplitude-consistent.
        """
        wav_16k_np, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        wav_16k_cpu = torch.from_numpy(wav_16k_np).float().unsqueeze(0)

        if self.model_sr == 16000:
            wav_model_cpu = wav_16k_cpu.clone()
        else:
            wav_model_np, _ = librosa.load(str(wav_path), sr=self.model_sr, mono=True)
            wav_model_cpu = torch.from_numpy(wav_model_np).float().unsqueeze(0)

        if self.peak_normalize:
            wav_16k_cpu, wav_model_cpu = self._peak_normalize_pair(
                wav_16k=wav_16k_cpu,
                wav_model=wav_model_cpu,
                peak_target=self.peak_target,
            )

        return wav_16k_cpu, wav_model_cpu

    def _make_condition(
        self,
        wav_16k: torch.Tensor,
        y_dtype: torch.dtype,
        batch_size: int,
    ) -> torch.Tensor | None:
        emb = self.speaker_encoder(wav_16k)
        return emb.view(batch_size, -1).to(device=self.device, dtype=y_dtype)

    def _compute_spec_lengths(self, wav_lengths: torch.Tensor) -> torch.Tensor:
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

    def _validate_spec_lengths(
        self,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        wav_lengths: torch.Tensor,
    ) -> None:
        """
        Sanity-check length formula for the current STFT convention.

        This is especially useful for TIMIT WBE because frame-to-time conversion
        assumes y_lengths and y.size(2) are in the same frame coordinate.
        """
        if y.size(0) == 1:
            actual = int(y.size(2))
            computed = int(y_lengths[0].item())

            if computed != actual:
                raise RuntimeError(
                    "Spec length mismatch in AlignerInputMaker. "
                    + f"computed y_lengths={computed}, actual y.size(2)={actual}, "
                    + f"wav_lengths={wav_lengths.tolist()}, "
                    + f"sr={self.audio_config.sr}, "
                    + f"hop={self.audio_config.hop_length}, "
                    + f"n_fft={self.audio_config.n_fft}"
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

    @staticmethod
    def _as_path_list(paths: str | Path | list[str | Path]) -> list[Path]:
        if isinstance(paths, (str, Path)):
            return [Path(paths)]
        return [Path(path) for path in paths]

    @staticmethod
    def _as_script_list(scripts: list[str]) -> list[str]:
        if isinstance(scripts, str):
            raise TypeError("scripts must be a sequence of str, not a bare str")

        return list(scripts)

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
