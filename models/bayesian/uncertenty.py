"""
uncertainty.py
Uncertainty metrics for BayesianNN output in AIPlasma framework.

Lives in: models/bayesian/uncertainty.py
"""

import torch
from torch import Tensor


class UncertaintyMetrics:
    """
    Utility class for computing uncertainty metrics from BayesianNN output.

    All methods are static — no instantiation needed:
        ece = UncertaintyMetrics.calibration_error(mean, std, targets)

    Methods:
        epistemic()        — model uncertainty (variance across MC samples)
        aleatoric()        — data uncertainty (residual after model variance)
        calibration_error() — Expected Calibration Error (ECE)
    """

    @staticmethod
    def epistemic(samples: Tensor) -> Tensor:
        """
        Computes epistemic (model) uncertainty.

        Epistemic uncertainty reflects what the model does not know —
        it decreases as more training data is provided.
        Estimated as the variance across MC Dropout samples.

        Args:
            samples: Stacked MC samples, shape (n_samples, N, output_dim).

        Returns:
            Epistemic uncertainty, shape (N, output_dim).
            Higher values indicate regions where the model is uncertain.
        """
        return samples.var(dim=0)  # variance across MC samples

    @staticmethod
    def aleatoric(samples: Tensor, targets: Tensor) -> Tensor:
        """
        Computes aleatoric (data) uncertainty.

        Aleatoric uncertainty reflects noise inherent in the data —
        it cannot be reduced by adding more training data.
        Estimated as the mean squared residual after subtracting
        epistemic variance.

        Args:
            samples: Stacked MC samples, shape (n_samples, N, output_dim).
            targets: Ground truth values, shape (N, output_dim).

        Returns:
            Aleatoric uncertainty, shape (N, output_dim).
            Clamped to >= 0 to avoid negative values from estimation noise.
        """
        mean        = samples.mean(dim=0)               # (N, output_dim)
        total_var   = ((samples - targets) ** 2).mean(dim=0)
        epistemic   = UncertaintyMetrics.epistemic(samples)
        aleatoric   = total_var - epistemic
        return torch.clamp(aleatoric, min=0.0)

    @staticmethod
    def calibration_error(
        pred_mean: Tensor,
        pred_std:  Tensor,
        targets:   Tensor,
        n_bins:    int = 10,
    ) -> float:
        """
        Computes Expected Calibration Error (ECE).

        ECE measures how well the predicted uncertainty corresponds
        to the actual prediction error. A well-calibrated model has
        ECE close to 0.

        Method:
            1. Compute normalized residuals: z = |targets - mean| / std
            2. Bin z values into n_bins
            3. For each bin: compare expected vs actual coverage
            4. ECE = weighted average of bin-wise calibration errors

        Args:
            pred_mean: Mean predictions, shape (N, output_dim).
            pred_std:  Predicted std, shape (N, output_dim).
            targets:   Ground truth values, shape (N, output_dim).
            n_bins:    Number of calibration bins. Default: 10.

        Returns:
            ECE as a float. Lower is better. Perfect calibration = 0.0.
        """
        # Avoid division by zero
        std_safe = torch.clamp(pred_std, min=1e-8)

        # Normalized absolute residuals
        z = torch.abs(targets - pred_mean) / std_safe  # (N, output_dim)
        z = z.flatten()

        # Bin edges from 0 to max z
        bin_edges = torch.linspace(0.0, z.max().item(), n_bins + 1)

        ece        = 0.0
        n_total    = z.shape[0]

        for i in range(n_bins):
            lo   = bin_edges[i].item()
            hi   = bin_edges[i + 1].item()
            mask = (z >= lo) & (z < hi)
            n_bin = mask.sum().item()

            if n_bin == 0:
                continue

            # Expected: fraction of points that should fall in this bin
            expected = (hi - lo) / bin_edges[-1].item()

            # Actual: fraction of points that fall in this bin
            actual = n_bin / n_total

            ece += abs(actual - expected) * (n_bin / n_total)

        return float(ece)