import os

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
from numpy.typing import NDArray
from silero_vad import get_speech_timestamps, load_silero_vad

from ..preprocess.config import PreprocessConfig
from .utils.freq_filter import lowpass_filter

SR_16K = 16000


class AudioPreprocessor:
    def __init__(self, config: PreprocessConfig):
        # resample parameters
        self.resample_sr = config.resample_sr

        # silence trimming parameters
        self.silence_trim = config.silence_trim
        self.denoise_before_vad = config.denoise_before_vad
        self.lpf_before_vad = config.lpf_before_vad
        self.lpf_cutoff_freq = config.lpf_cutoff_freq

        self.silence_margin_sec = config.silence_margin_sec

        # rms-normalizing parameters
        self.rms_normalize = config.rms_normalize
        self.rms_target = config.rms_target

        # silero model
        self.silero_model = load_silero_vad()

    def process_audio(self, input_path: str, output_path: str) -> bool:
        """
        Processes a single audio file: load, DSP pipeline, and save as WAV.
        Returns:
            bool: True if processed and saved, False if skipped (e.g. empty after trimming)
        """
        try:
            wav_np, sr = librosa.load(input_path, sr=None)
            wav_np = self._process_unit(wav_np, int(sr))

            if len(wav_np) > 0:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self._save_audio(
                    path=output_path,
                    y=wav_np,
                    sr=self.resample_sr,
                )
                return True
            else:
                return False
        except Exception as e:
            raise RuntimeError(f"Failed to process {input_path}: {e}")

    def _process_unit(
        self,
        y: NDArray[np.float32],
        sr: int,
    ) -> NDArray[np.float32]:

        # 1. Target Sample Rate
        y_target = (
            librosa.resample(
                y,
                orig_sr=sr,
                target_sr=self.resample_sr,
            )
            if sr != self.resample_sr
            else y
        )

        # 2. Silence Trimming
        if self.silence_trim:
            y_target = self._trim_silence(
                y_target,
                sr=self.resample_sr,
            )

            if len(y_target) == 0:
                return y_target

        # 3. RMS Normalization
        if self.rms_normalize:
            y_target = self._rms_normalize(
                y_target,
                rms_target=self.rms_target,
            )

        # 4. Add Silence Margin
        if self.silence_margin_sec > 0.0:
            y_target = self._add_silence_margin(
                y_target,
                sr=self.resample_sr,
                margin_sec=self.silence_margin_sec,
            )

        return y_target

    def _trim_silence(self, y: NDArray[np.float32], sr: int) -> NDArray[np.float32]:
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=SR_16K) if sr != SR_16K else y
        y_16k_processed = y_16k.copy()

        if self.denoise_before_vad:
            wav_16k_rms = np.sqrt(np.mean(y_16k_processed**2))

            if wav_16k_rms > 1e-7:
                wav_16k_denoised = nr.reduce_noise(
                    y=y_16k_processed,
                    sr=SR_16K,
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
                sr=SR_16K,
                cutoff_freq=self.lpf_cutoff_freq,
                order=4,
            )

        wav_tensor = torch.from_numpy(y_16k_processed)
        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            self.silero_model,
            sampling_rate=SR_16K,
            threshold=0.1,
            neg_threshold=0.15,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            speech_pad_ms=0,
            return_seconds=False,
            window_size_samples=512,
        )

        if not speech_timestamps:
            return np.array([], dtype=np.float32)

        ratio = sr / SR_16K
        first_start = speech_timestamps[0]["start"]
        last_end = speech_timestamps[-1]["end"]

        start_idx = int(first_start * ratio)
        end_idx = int(last_end * ratio)

        return y[start_idx:end_idx]

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
        return np.concatenate([silence, y, silence])

    @staticmethod
    def _rms_normalize(
        y: NDArray[np.float32],
        rms_target: float,
    ) -> NDArray[np.float32]:
        current_rms = np.sqrt(np.mean(y**2))

        if current_rms < 1e-7:
            return y

        y_norm = y * (rms_target / current_rms)
        y_norm = np.clip(y_norm, -1.0, 1.0)
        return y_norm

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
