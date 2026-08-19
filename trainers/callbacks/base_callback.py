from abc import ABC



# ════════════════════════════════════════════════════════════════════════════
# 1. Callback — base class
# Lives in: trainers/callbacks/base_callback.py
# ════════════════════════════════════════════════════════════════════════════

class Callback(ABC):
    """
    Abstract base class for all AIPlasma callbacks.

    All methods have empty default implementations — subclasses
    override only the events they need.

    Events fired by BaseTrainer.fit():
        on_train_start  — once before training begins
        on_epoch_start  — at the beginning of each epoch
        on_epoch_end    — at the end of each epoch
        on_train_end    — once after training completes

    Example:
        class MyCallback(Callback):
            def on_epoch_end(self, trainer, epoch, history, **kwargs):
                print(f"Epoch {epoch} done.")
    """

    def on_train_start(self, trainer, **kwargs) -> None:
        pass

    def on_epoch_start(self, trainer, epoch: int, **kwargs) -> None:
        pass

    def on_epoch_end(self, trainer, epoch: int, history, **kwargs) -> None:
        pass

    def on_train_end(self, trainer, history, **kwargs) -> None:
        pass

