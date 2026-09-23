"""
rl_trainer.py
Reinforcement Learning Trainer for plasma control in AIPlasma.

Lives in: trainers/rl_trainer.py

Uses REINFORCE (policy gradient) algorithm.
Inherits BaseTrainer — reuses fit(), callbacks, and monitoring (ADR-014).
"""

import torch
from torch import Tensor
from typing import Optional

from core.base_trainer import BaseTrainer, TrainerConfig
from core.base_model import PhysicsModel
from core.base_problem import PhysicsProblem
from core.base_solver import PhysicsSolver
from data.preprocessing.feature_pipeline import FeatureBatch
from trainers.environments.plasma_env import PlasmaEnvironment


class RLTrainer(BaseTrainer):
    """
    REINFORCE policy gradient trainer for plasma control.

    Maps RL concepts to BaseTrainer interface (ADR-014):
        state  = FeatureBatch  — plasma diagnostics from environment
        action = policy(state) — control parameters
        reward = env.step(action) — confinement feedback
        loss   = -reward.mean()   — policy gradient loss

    train_step() ignores the batch argument from fit() and uses
    env.state() directly — batch parameter kept for interface consistency.

    Args:
        model:       Policy network (PhysicsModel subclass).
                     Maps state (FeatureBatch) → action (Tensor).
        problem:     PhysicsProblem defining physics constraints.
        solver:      PhysicsSolver instance.
        config:      TrainerConfig with training hyperparameters.
        env:         PlasmaEnvironment instance.
        lr:          Learning rate for Adam optimizer. Default: 1e-3.
        n_steps:     Number of environment steps per train_step(). Default: 10.
        gamma:       Discount factor for returns. Default: 0.99.
        entropy_coef: Entropy regularization coefficient. Default: 0.01.

    Example:
        env     = SyntheticPlasmaEnv(n_points=50)
        policy  = BasePINN(input_dim=2, output_dim=50)
        trainer = RLTrainer(
            model=policy, problem=problem, solver=solver,
            config=TrainerConfig(max_epochs=100),
            env=env,
        )
        history = trainer.fit(train_batches=[env.reset()],
                              val_batches=[env.reset()])
    """

    callbacks: list = []

    def __init__(
        self,
        model:        PhysicsModel,
        problem:      PhysicsProblem,
        solver:       PhysicsSolver,
        config:       TrainerConfig,
        env:          PlasmaEnvironment,
        lr:           float = 1e-3,
        n_steps:      int   = 10,
        gamma:        float = 0.99,
        entropy_coef: float = 0.01,
    ):
        super().__init__(model, problem, solver, config)

        self.env          = env
        self.n_steps      = n_steps
        self.gamma        = gamma
        self.entropy_coef = entropy_coef

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

        # Episode history for monitoring
        self._episode_rewards: list[float] = []

    # ── Abstract implementations ─────────────────────────────────────────────

    def train_step(self, batch: FeatureBatch) -> Tensor:
        """
        Single REINFORCE policy gradient step.

        Runs n_steps in the environment, collects rewards,
        and updates policy using discounted returns.

        batch argument is unused — env.state() provides the state.
        Kept for interface consistency with BaseTrainer (ADR-014).

        Returns:
            Policy loss tensor (negative mean discounted return).
        """
        self.optimizer.zero_grad()

        log_probs = []
        rewards   = []

        # Collect n_steps of experience
        for _ in range(self.n_steps):
            state  = self.env.state()
            state  = self._to_device(state)

            # Policy forward: state → action logits
            output = self.model.forward(state)
            action_logits = output.pred  # (N, action_dim)

            # Sample action from Gaussian policy
            action_mean = action_logits.mean(dim=0)  # (action_dim,)
            action_std  = torch.ones_like(action_mean) * 0.1
            dist        = torch.distributions.Normal(action_mean, action_std)
            action      = dist.sample()
            log_prob    = dist.log_prob(action).sum()

            # Step environment
            reward = self.env.step(action.detach().cpu())

            log_probs.append(log_prob)
            rewards.append(reward.item())

            if self.env.is_terminal():
                self.env.reset()
                break

        # Compute discounted returns
        returns    = self._compute_returns(rewards)
        returns    = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns    = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE loss: -sum(log_prob * return)
        log_probs_t  = torch.stack(log_probs)
        policy_loss  = -(log_probs_t * returns).mean()

        # Entropy regularization — encourages exploration
        entropy_loss = -self.entropy_coef * dist.entropy().mean()
        loss         = policy_loss + entropy_loss

        loss.backward()
        self.optimizer.step()

        # Track episode rewards
        self._episode_rewards.append(sum(rewards))

        return loss

    def val_step(self, batch: FeatureBatch) -> Tensor:
        """
        Evaluates policy by running one full episode.

        Runs until terminal state or n_steps, returns
        negative mean reward as validation loss.
        """
        total_reward = 0.0
        steps        = 0

        self.env.reset()

        for _ in range(self.n_steps):
            state  = self.env.state()
            state  = self._to_device(state)
            output = self.model.forward(state)

            action_mean = output.pred.mean(dim=0)
            reward      = self.env.step(action_mean.detach().cpu())
            total_reward += reward.item()
            steps        += 1

            if self.env.is_terminal():
                break

        # Return negative mean reward as loss (lower = better policy)
        return torch.tensor(-total_reward / max(steps, 1))

    # ── RL utilities ─────────────────────────────────────────────────────────

    def _compute_returns(self, rewards: list[float]) -> list[float]:
        """
        Computes discounted returns for each timestep.

        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...

        Args:
            rewards: List of rewards collected during episode.

        Returns:
            List of discounted returns, same length as rewards.
        """
        returns = []
        G       = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def episode_rewards(self) -> list[float]:
        """
        Returns total reward per training episode.

        Useful for monitoring policy improvement over time.
        """
        return self._episode_rewards