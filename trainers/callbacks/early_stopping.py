from trainers.callbacks.base_callback import Callback


class EarlyStopping(Callback):
    """
    Stops training when validation loss stops improving.

    Sets trainer._stop_training = True when the monitored metric
    has not improved by at least min_delta for patience epochs.
    BaseTrainer.fit() checks this flag after each epoch.

    Args:
        patience:  Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as improvement.
        monitor:   Metric to monitor. Default: 'val_loss'.
        verbose:   If True, prints a message when stopping.

    Example:
        early_stop = EarlyStopping(patience=10, min_delta=1e-4)
    """

    def __init__(
            self,
            patience: int = 10,
            min_delta: float = 1e-4,
            monitor: str = "val_loss",
            verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.verbose = verbose
        self._best_loss = float("inf")
        self._counter = 0

    def on_train_start(self, trainer, **kwargs) -> None:
        """Initializes stop flag on trainer."""
        trainer._stop_training = False

    def on_epoch_end(self, trainer, epoch: int, history, **kwargs) -> None:
        """
        Checks if training should stop.

        If val_loss has not improved by min_delta for patience epochs,
        sets trainer._stop_training = True.
        """
        if not history.val_loss:
            return

        current_loss = history.val_loss[-1]

        if current_loss < self._best_loss - self.min_delta:
            self._best_loss = current_loss
            self._counter = 0
        else:
            self._counter += 1

        if self._counter >= self.patience:
            trainer._stop_training = True
            if self.verbose:
                print(
                    f"\n  [EarlyStopping] Stopping at epoch {epoch}. "
                    f"No improvement for {self.patience} epochs."
                )
