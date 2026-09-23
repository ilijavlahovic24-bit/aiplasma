"""
plasma_env.py
Abstract PlasmaEnvironment for RL-based plasma control in AIPlasma.

Lives in: trainers/environments/plasma_env.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import torch
from torch import Tensor

from data.preprocessing.feature_pipeline import FeatureBatch


# ════════════════════════════════════════════════════════════════════════════
# 1. PlasmaEnvironment — abstract base
# ════════════════════════════════════════════════════════════════════════════

class PlasmaEnvironment(ABC):
    """
    Abstract environment for plasma control RL problems.

    Maps plasma control to standard RL concepts:
        state  = FeatureBatch  — current plasma diagnostics
        action = Tensor        — control parameters (B field, heating, etc.)
        reward = Tensor        — scalar feedback from simulation

    Users subclass PlasmaEnvironment to define their specific
    plasma control problem. The simulator can be analytical,
    numerical, or a surrogate model.

    Example:
        class TokamakControlEnv(PlasmaEnvironment):
            def step(self, action):
                new_state = self._simulate(action)
                reward    = self._compute_reward(new_state)
                return reward

            def reset(self):
                return self._initial_state()

            def state(self):
                return self._current_state
    """

    @abstractmethod
    def step(self, action: Tensor) -> Tensor:
        """
        Applies action to the environment and returns reward.

        Args:
            action: Control parameters, shape (action_dim,).
                    e.g. [B_field, heating_power, current_profile]

        Returns:
            Scalar reward tensor. Higher = better plasma confinement.
        """
        ...

    @abstractmethod
    def reset(self) -> FeatureBatch:
        """
        Resets environment to initial state.

        Returns:
            Initial FeatureBatch representing plasma state at t=0.
        """
        ...

    @abstractmethod
    def state(self) -> FeatureBatch:
        """
        Returns current environment state.

        Returns:
            Current FeatureBatch representing plasma diagnostics.
        """
        ...

    def action_dim(self) -> int:
        """
        Number of control parameters (action space dimension).
        Override to specify action space size. Default: 1.
        """
        return 1

    def is_terminal(self) -> bool:
        """
        Returns True if current state is terminal (e.g. disruption).
        Override for episodic environments. Default: False.
        """
        return False

    def describe(self) -> str:
        return f"{self.__class__.__name__}(action_dim={self.action_dim()})"


# ════════════════════════════════════════════════════════════════════════════
# 2. Synthetic PlasmaEnvironment for testing
# ════════════════════════════════════════════════════════════════════════════

class SyntheticPlasmaEnv(PlasmaEnvironment):
    """
    Synthetic plasma environment for testing and benchmarking.

    Simulates a simple 1D plasma confinement problem where:
        state  = temperature profile u(x, t)
        action = heating power applied at each point
        reward = -mean(|u - u_target|²)  (negative MSE from target profile)

    Not physically accurate — intended for testing RLTrainer only.

    Args:
        n_points:  Number of spatial points. Default: 50.
        u_target:  Target temperature profile. Default: sin(πx).
    """

    def __init__(self, n_points: int = 50):
        self.n_points = n_points
        self._x       = torch.linspace(0, 1, n_points)
        self._t       = torch.zeros(n_points)
        self._u       = torch.sin(torch.pi * self._x)   # initial profile
        self._u_target = torch.sin(torch.pi * self._x)  # target profile

    def step(self, action: Tensor) -> Tensor:
        """
        Applies heating action and returns confinement reward.
        Reward = -MSE(current_profile, target_profile).
        """
        # Simple dynamics: action perturbs the profile
        self._u = self._u + 0.01 * action.squeeze()
        self._u = torch.clamp(self._u, 0.0, 2.0)
        self._t = self._t + 0.01

        reward = -torch.mean((self._u - self._u_target) ** 2)
        return reward.unsqueeze(0)

    def reset(self) -> FeatureBatch:
        self._u = torch.sin(torch.pi * self._x)
        self._t = torch.zeros(self.n_points)
        return self.state()

    def state(self) -> FeatureBatch:
        coords = torch.stack([self._x, self._t], dim=1)
        fields = self._u.unsqueeze(1)
        bm     = (self._x < 0.05) | (self._x > 0.95)
        return FeatureBatch(
            coords=coords,
            fields=fields,
            boundary_mask=bm,
            collocation_mask=~bm,
            physics_params={"heating": 0.0},
            fidelity_level=0,
            fidelity_weight=1.0,
        )

    def action_dim(self) -> int:
        return self.n_points