import torch
from torch import nn
from torch.nn import functional as F

from hubbard_ac.models.mlp import trapezoid_weights


class AnalyticContinuationLoss(nn.Module):
    def __init__(
        self,
        omega: torch.Tensor,
        tau_fraction: torch.Tensor,
        beta_values: torch.Tensor,
        lambda_spectrum: float = 1.0,
        lambda_green: float = 1.0,
    ):
        super().__init__()

        omega = torch.as_tensor(
            omega,
            dtype=torch.float32,
        )

        tau_fraction = torch.as_tensor(
            tau_fraction,
            dtype=torch.float32,
        )

        beta_values = torch.as_tensor(
            beta_values,
            dtype=torch.float32,
        )

        weights = trapezoid_weights(
            omega
        )

        self.register_buffer(
            "omega",
            omega,
        )

        self.register_buffer(
            "tau_fraction",
            tau_fraction,
        )

        self.register_buffer(
            "beta_values",
            beta_values,
        )

        self.register_buffer(
            "integration_weights",
            weights,
        )

        kernels = []

        for beta in beta_values:
            tau = beta * tau_fraction

            exponent = (
                -tau[:, None] * omega[None, :]
                - torch.logaddexp(
                    torch.zeros(
                        (),
                        dtype=omega.dtype,
                    ),
                    -beta * omega[None, :],
                )
            )

            kernel = -torch.exp(
                exponent
            )

            kernel = (
                kernel
                * weights[None, :]
            )

            kernels.append(kernel)

        self.register_buffer(
            "kernels",
            torch.stack(
                kernels,
                dim=0,
            ),
        )

        self.lambda_spectrum = float(
            lambda_spectrum
        )

        self.lambda_green = float(
            lambda_green
        )

    def beta_indices(
        self,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        differences = torch.abs(
            beta[:, None]
            - self.beta_values[None, :]
        )

        indices = torch.argmin(
            differences,
            dim=1,
        )

        selected = self.beta_values[
            indices
        ]

        if not torch.allclose(
            selected,
            beta,
            atol=1.0e-5,
            rtol=0.0,
        ):
            raise ValueError(
                "batch contains beta not present in kernel bank"
            )

        return indices

    def reconstruct_green(
        self,
        spectrum: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        indices = self.beta_indices(
            beta
        )

        output = torch.empty(
            (
                spectrum.shape[0],
                len(self.tau_fraction),
            ),
            dtype=spectrum.dtype,
            device=spectrum.device,
        )

        for kernel_index in range(
            len(self.beta_values)
        ):
            mask = (
                indices == kernel_index
            )

            if not torch.any(mask):
                continue

            output[mask] = (
                spectrum[mask]
                @ self.kernels[
                    kernel_index
                ].T
            )

        return output

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        g_input: torch.Tensor,
        beta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        spectrum_loss = F.mse_loss(
            prediction,
            target,
        )

        g_prediction = self.reconstruct_green(
            prediction,
            beta,
        )

        green_loss = F.mse_loss(
            g_prediction,
            g_input,
        )

        total = (
            self.lambda_spectrum
            * spectrum_loss
            + self.lambda_green
            * green_loss
        )

        return {
            "loss": total,
            "spectrum_loss": spectrum_loss,
            "green_loss": green_loss,
        }
