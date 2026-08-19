from trainers.callbacks.base_callback import Callback
from typing import Optional
import torch
import os

class CheckPointing(Callback):
    """
    Saves the model when validation loss improves.

    Args:
        save_dir:      Directory where checkpoints are saved.
        monitor:       Metric to monitor. Default: 'val_loss'.
        save_best_only: If True, only saves when monitored metric improves.
                        If False, saves every checkpoint_frequency epochs.
        checkpoint_frequency: Epochs between saves when save_best_only=False.
        verbose:       If True, prints a message when saving.

    Example:
        checkpoint = CheckPointing(save_dir="checkpoints/", verbose=True)
    """

    def __init__(
            self,
            save_dir: str,
            monitor: str = "val_loss",
            save_best_only: bool = True,
            checkpoint_frequency: int = 10,
            verbose: bool = False,
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.checkpoint_frequency = checkpoint_frequency
        self.verbose = verbose
        self._best_loss = float("inf")

        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, trainer, epoch: int, history, **kwargs) -> None:
        """
        Saves model checkpoint if conditions are met.

        For save_best_only=True: saves when val_loss improves.
        For save_best_only=False: saves every checkpoint_frequency epochs.
        """
        if self.save_best_only:
            if not history.val_loss:
                return

            current_loss = history.val_loss[-1]
            if current_loss < self._best_loss:
                self._best_loss = current_loss
                self._save(trainer, epoch, current_loss)
        else:
            if epoch % self.checkpoint_frequency == 0:
                loss = history.train_loss[-1] if history.train_loss else 0.0
                self._save(trainer, epoch, loss)

    def _save(self, trainer, epoch: int, loss: float) -> None:
        """
        Saves model state dict to disk.

        File naming: checkpoint_epoch_{epoch}_loss_{loss:.4f}.pt
        """
        filename = f"checkpoint_epoch_{epoch:04d}_loss_{loss:.6f}.pt"
        path = os.path.join(self.save_dir, filename)

        torch.save({
            "epoch": epoch,
            "loss": loss,
            "model_state": trainer.model.state_dict(),
        }, path)

        if self.verbose:
            print(f"  [CheckPointing] Saved checkpoint: {path}")
