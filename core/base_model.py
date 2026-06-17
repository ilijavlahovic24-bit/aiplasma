import sys
print(sys.path)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from data.preprocessing.feature_pipeline import FeatureBatch


@dataclass
class ModelOutput:
    pred: Tensor
    uncertainty: Optional[Tensor] = None
    aux: dict = field(default_factory=dict)

class PhysicsModel(ABC):

    # ── Korisnik MORA da implementira ───────────────────────────

    @abstractmethod
    def forward(self, batch: FeatureBatch) -> ModelOutput:
        """Predikcija polja na zadatim koordinatama."""
        ...

    # ── Framework pruža ─────────────────────────────────────────

    def predict(self, batch: FeatureBatch) -> ModelOutput:
        """Inference bez gradijenta."""
        with torch.no_grad():
            return self.forward(batch)

    def preprocess(self, batch: FeatureBatch) -> FeatureBatch:
        """Opcioni hook za model-specifičnu transformaciju."""
        return batch

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def summary(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"{self.__class__.__name__}: {total:,} parameters"

print("Model")