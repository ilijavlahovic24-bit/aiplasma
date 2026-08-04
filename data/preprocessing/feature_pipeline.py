#from data import input_pipeline, input_pipeline_multimer
from dataclasses import dataclass
from torch import Tensor

@dataclass
class FeatureBatch:
    coords: Tensor            # (N, d)
    fields: Tensor            # (N, f)
    boundary_mask: Tensor     # (N,) bool
    collocation_mask: Tensor  # (N,) bool
    physics_params: dict      # Re, beta, magnetic field strength
    fidelity_level: int       # 0 = coarse, N = fine
    fidelity_weight: float    # how much this level affects loss-u