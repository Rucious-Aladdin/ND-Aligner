import os

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from silero_vad import get_speech_timestamps, load_silero_vad

from ..audio.utils.freq_filter import lowpass_filter
from ..config.preprocess.preprocess_config import PreprocessConfigs


class AudioPreprocessor:
    def __init__(self, config: PreprocessConfigs):
        # resample parameters
        self.resample_sr = config.resample_sr

        # silence trimming parameters
        self.silence_trim = config.silence_trim
        self.denoise_before_vad = config.denoise_before_vad
        self.lpf_before_vad = config.lpf_before_vad
        self.lpf_cutoff_freq = config.lpf_cutoff_freq
        self.silence_margin_sec = config.silence_margin_sec

        # peak-normalizing parameters
        self.peak_normalize = config.peak_normalize
        self.peak_target = config.peak_target

        # silero model
        self.silero_model = load_silero_vad()

    def process_audio(self, input_path: str, output_path: str) -> bool:
        """
        Processes a single audio file: load, DSP pipeline, and save as WAV.

        Returns:
            bool: True if processed and saved, False if skipped.
        """
        try:
            wav_np, sr = librosa.load(input_path, sr=None, mono=True)
            wav_np = self._process_unit(wav_np, int(sr))

            if len(wav_np) > 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self._save_audio(
                    path=output_path,
                    y=wav_np,
                    sr=self.resample_sr,
                )
                return True

            return False

        except Exception as e:
            raise RuntimeError(f"Failed to process {input_path}: {e}") from e

    def _process_unit(
        self,
        y: NDArray[np.float32],
        sr: int,
    ) -> NDArray[np.float32]:

        # 1. Target sample rate
        y_target = (
            librosa.resample(
                y,
                orig_sr=sr,
                target_sr=self.resample_sr,
            )
            if sr != self.resample_sr
            else y
        ).astype(np.float32)

        if len(y_target) == 0:
            return y_target

        # 2. Peak normalization before VAD.
        # This makes Silero VAD less sensitive to input recording gain.
        if self.peak_normalize:
            y_target = self._peak_normalize(
                y=y_target,
                peak_target=self.peak_target,
            )

        # 3. Silence trimming
        if self.silence_trim:
            y_target = self._trim_silence(
                y_target,
                sr=self.resample_sr,
            )

            if len(y_target) == 0:
                return y_target

        # 4. Final peak normalization.
        # If the original global peak was outside the retained speech span,
        # trimming may reduce the final peak. Normalize again so saved audio
        # consistently has the requested peak scale.
        if self.peak_normalize:
            y_target = self._peak_normalize(
                y=y_target,
                peak_target=self.peak_target,
            )

        # 5. Add silence margin
        if self.silence_margin_sec > 0.0:
            y_target = self._add_silence_margin(
                y_target,
                sr=self.resample_sr,
                margin_sec=self.silence_margin_sec,
            )

        return y_target.astype(np.float32)

    def _trim_silence(self, y: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        speech_span = self._get_speech_span(y, sr=sr)

        if speech_span is None:
            return np.array([], dtype=np.float32)

        start_idx, end_idx = speech_span
        return y[start_idx:end_idx].astype(np.float32)

    def _get_speech_span(
        self,
        y: NDArray[np.float32],
        sr: int,
    ) -> tuple[int, int] | None:
        """
        Detect non-silence speech span using Silero VAD.

        Returns:
            (start_idx, end_idx) in the sampling rate of y.
            None if no speech is detected.
        """
        if len(y) == 0:
            return None

        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16_000) if sr != 16_000 else y
        y_16k_processed = y_16k.astype(np.float32).copy()

        if self.denoise_before_vad:
            wav_16k_rms = np.sqrt(np.mean(y_16k_processed**2))

            if wav_16k_rms > 1e-7:
                wav_16k_denoised = nr.reduce_noise(
                    y=y_16k_processed,
                    sr=16_000,
                    prop_decrease=0.9,
                    n_fft=1024,
                    win_length=1024,
                    hop_length=320,
                )

                denoised_rms = np.sqrt(np.mean(wav_16k_denoised**2))
                if denoised_rms > 1e-7:
                    gain = wav_16k_rms / denoised_rms
                    y_16k_processed = wav_16k_denoised * gain

        if self.lpf_before_vad:
            y_16k_processed = lowpass_filter(
                x=y_16k_processed,
                sr=16_000,
                cutoff_freq=self.lpf_cutoff_freq,
                order=4,
            )

        wav_tensor = torch.from_numpy(y_16k_processed.astype(np.float32))

        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            self.silero_model,
            sampling_rate=16_000,
            threshold=0.5,
            neg_threshold=None,  # type: ignore
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            return_seconds=False,
            window_size_samples=512,
        )

        if not speech_timestamps:
            return None

        ratio = sr / 16_000
        first_start = int(speech_timestamps[0]["start"])
        last_end = int(speech_timestamps[-1]["end"])

        start_idx = int(first_start * ratio)
        end_idx = int(last_end * ratio)

        start_idx = max(0, min(start_idx, len(y)))
        end_idx = max(start_idx, min(end_idx, len(y)))

        if end_idx <= start_idx:
            return None

        return start_idx, end_idx

    @staticmethod
    def _peak_normalize(
        y: NDArray[np.float32],
        peak_target: float,
        eps: float = 1e-7,
    ) -> NDArray[np.float32]:
        if len(y) == 0:
            return y

        peak = float(np.max(np.abs(y)))

        if peak < eps:
            return y.astype(np.float32)

        gain = float(peak_target) / peak
        y_norm = y.astype(np.float32) * gain
        y_norm = np.clip(y_norm, -1.0, 1.0)

        return y_norm.astype(np.float32)

    @staticmethod
    def _add_silence_margin(
        y: NDArray[np.float32],
        sr: int,
        margin_sec: float,
    ) -> NDArray[np.float32]:
        margin_samples = int(sr * margin_sec)
        if margin_samples == 0:
            return y

        silence = np.zeros(margin_samples, dtype=np.float32)
        return np.concatenate([silence, y, silence]).astype(np.float32)

    @staticmethod
    def _save_audio(
        path: str,
        y: NDArray[np.float32],
        sr: int,
    ):
        sf.write(
            file=path,
            data=y,
            samplerate=sr,
        )
