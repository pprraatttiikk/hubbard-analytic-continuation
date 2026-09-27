import math

import torch
from torch import nn
from torch.nn import functional as F


def trapezoid_weights(grid: torch.Tensor) -> torch.Tensor:
    if grid.ndim != 1:
        raise ValueError("grid must be one-dimensional")

    if len(grid) < 2:
        raise ValueError("grid must contain at least two points")

    dx = grid[1:] - grid[:-1]

    if torch.any(dx <= 0):
        raise ValueError("grid must be strictly increasing")

    weights = torch.empty_like(grid)

    weights[0] = 0.5 * dx[0]
    weights[-1] = 0.5 * dx[-1]

    if len(grid) > 2:
        weights[1:-1] = 0.5 * (
            dx[:-1] + dx[1:]
        )

    return weights


class ConditioningNormalizer(nn.Module):
    def __init__(
        self,
        u_max: float = 12.0,
        beta_max: float = 50.0,
        noise_floor: float = 1.0e-8,
    ):
        super().__init__()

        self.u_max = float(u_max)
        self.beta_max = float(beta_max)
        self.noise_floor = float(noise_floor)

    def forward(
        self,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        if conditioning.shape[-1] != 4:
            raise ValueError(
                "conditioning must contain U, beta, k, sigma"
            )

        U = conditioning[..., 0]
        beta = conditioning[..., 1]
        k = conditioning[..., 2]
        sigma = conditioning[..., 3]

        U_scaled = U / self.u_max
        beta_scaled = beta / self.beta_max
        k_scaled = k / (2.0 * math.pi)

        noise_scaled = (
            torch.log10(sigma + self.noise_floor)
            + 8.0
        ) / 5.0

        return torch.stack(
            (
                U_scaled,
                beta_scaled,
                k_scaled,
                noise_scaled,
            ),
            dim=-1,
        )


class HubbardMLP(nn.Module):
    def __init__(
        self,
        omega: torch.Tensor,
        n_tau: int = 240,
        hidden_dim: int = 256,
        parameter_dim: int = 64,
        dropout: float = 0.05,
        u_max: float = 12.0,
        beta_max: float = 50.0,
    ):
        super().__init__()

        omega = torch.as_tensor(
            omega,
            dtype=torch.float32,
        )

        self.register_buffer(
            "omega",
            omega,
        )

        self.register_buffer(
            "integration_weights",
            trapezoid_weights(omega),
        )

        self.conditioning_normalizer = (
            ConditioningNormalizer(
                u_max=u_max,
                beta_max=beta_max,
            )
        )

        self.g_encoder = nn.Sequential(
            nn.Linear(n_tau, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.parameter_encoder = nn.Sequential(
            nn.Linear(4, parameter_dim),
            nn.GELU(),
            nn.Linear(parameter_dim, parameter_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_dim + parameter_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, len(omega)),
        )

    def forward(
        self,
        g_input: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self.conditioning_normalizer(
            conditioning
        )

        g_features = self.g_encoder(
            g_input
        )

        parameter_features = self.parameter_encoder(
            parameters
        )

        features = torch.cat(
            (
                g_features,
                parameter_features,
            ),
            dim=-1,
        )

        raw = self.decoder(features)

        spectrum = F.softplus(raw)

        normalization = torch.sum(
            spectrum
            * self.integration_weights,
            dim=-1,
            keepdim=True,
        )

        return spectrum / normalization
