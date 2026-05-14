import os

import torch
from matplotlib.figure import Figure
from torch.utils.tensorboard.writer import SummaryWriter

class TensorboardLogger:
    def __init__(self, log_dir: str, run_name: str):
        """
        Initializes the TensorBoard SummaryWriter.

        Args:
            log_dir (str): Base directory for logs.
            run_name (str): Name of the current run.
        """
        self.log_dir = os.path.join(log_dir, run_name)

        # Create directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard Logger initialized at: {self.log_dir}")

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "Train",
    ) -> None:
        """
        Logs scalar metrics (e.g., losses, learning rate) to TensorBoard.

        Args:
            metrics (Dict[str, float]): Dictionary of metric names and their values.
            step (int): Current global training step or epoch.
            prefix (str): Prefix for the metric group (e.g., 'Train' or 'Valid').
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def log_audio(
        self,
        tag: str,
        audio: torch.Tensor,
        step: int,
        sample_rate: int = 22050,
    ) -> None:
        """
        Logs generated or ground-truth audio to TensorBoard.

        Args:
            tag (str): The label for the audio (e.g., 'Valid/Predicted_Audio').
            audio (torch.Tensor): 1D or 2D audio tensor. Shape should be (T,) or (1, T).
            step (int): Current global step.
            sample_rate (int): Sampling rate of the audio.
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # Tensorboard expects (1, T) or (C, T)

        self.writer.add_audio(
            tag=tag,
            snd_tensor=audio,
            global_step=step,
            sample_rate=sample_rate,
        )

    def log_figure(
        self,
        tag: str,
        figure: Figure,
        step: int,
    ) -> None:
        """
        Logs a matplotlib figure (e.g., Mel-spectrogram, Attention map) to TensorBoard.

        Args:
            tag (str): The label for the figure (e.g., 'Valid/Attention_Map').
            figure (plt.Figure): The matplotlib figure object.
            step (int): Current global step.
        """
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """
        Logs text (e.g., scripts) to TensorBoard.

        Args:
            tag (str): The label for the text (e.g., 'Valid/Script').
            text (str): The text content to log.
            step (int): Current global step.
        """
        self.writer.add_text(tag=tag, text_string=text, global_step=step)

    def log_learning_rate(
        self,
        lr: float,
        step: int,
    ) -> None:
        """
        Logs the current learning rate.
        """
        self.writer.add_scalar("Train/Learning_Rate", lr, step)

    def close(self) -> None:
        """
        Closes the TensorBoard writer. Should be called at the end of training.
        """
        self.writer.close()
