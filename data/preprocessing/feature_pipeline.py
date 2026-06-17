#from data import input_pipeline, input_pipeline_multimer
from dataclasses import dataclass
from torch import Tensor

@dataclass
class FeatureBatch:
    coords: Tensor            # (N, d) — prostorno-vremenski ulaz za PINN
    fields: Tensor            # (N, f) — izmerene veličine
    boundary_mask: Tensor     # (N,) bool — koja tačka je BC
    collocation_mask: Tensor  # (N,) bool — interior tačke za PDE residual
    physics_params: dict      # Re, beta, magnetic field strength
    fidelity_level: int       # 0 = najgrublje, N = najfinije
    fidelity_weight: float    # koliko ovaj nivo doprinosi loss-u