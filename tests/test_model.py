import numpy as np
import torch

from hubbard_ac.models.losses import (
    AnalyticContinuationLoss,
)
from hubbard_ac.models.mlp import (
    ConditioningNormalizer,
    HubbardMLP,
    trapezoid_weights,
)


def test_trapezoid_weights():
    grid = torch.linspace(
        -2.0,
        2.0,
        101,
    )

    weights = trapezoid_weights(
        grid
    )

    assert torch.isclose(
        weights.sum(),
        torch.tensor(4.0),
        atol=1.0e-6,
    )


def test_conditioning_normalizer():
    normalizer = ConditioningNormalizer()

    conditioning = torch.tensor(
        [
            [12.0, 50.0, 2.0 * np.pi, 1.0e-3],
            [0.0, 10.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    output = normalizer(
        conditioning
    )

    assert torch.all(
        torch.isfinite(output)
    )

    assert torch.allclose(
        output[0],
        torch.ones(4),
        atol=1.0e-5,
    )

    assert torch.isclose(
        output[1, 0],
        torch.tensor(0.0),
    )

    assert torch.isclose(
        output[1, 3],
        torch.tensor(0.0),
        atol=1.0e-6,
    )


def test_model_output_shape():
    omega = torch.linspace(
        -10.0,
        10.0,
        401,
    )

    model = HubbardMLP(
        omega=omega,
        n_tau=40,
        hidden_dim=32,
        parameter_dim=8,
    )

    G = torch.randn(
        7,
        40,
    )

    conditioning = torch.tensor(
        [
            [4.0, 10.0, 0.0, 1.0e-4]
        ]
    ).repeat(
        7,
        1,
    )

    output = model(
        G,
        conditioning,
    )

    assert output.shape == (
        7,
        401,
    )


def test_model_output_is_positive_and_normalized():
    omega = torch.linspace(
        -10.0,
        10.0,
        401,
    )

    model = HubbardMLP(
        omega=omega,
        n_tau=40,
        hidden_dim=32,
        parameter_dim=8,
    )

    G = torch.randn(
        5,
        40,
    )

    conditioning = torch.tensor(
        [
            [4.0, 20.0, np.pi, 1.0e-3]
        ]
    ).repeat(
        5,
        1,
    )

    output = model(
        G,
        conditioning,
    )

    assert torch.all(
        output > 0.0
    )

    weights = trapezoid_weights(
        omega
    )

    integrals = torch.sum(
        output * weights,
        dim=1,
    )

    assert torch.allclose(
        integrals,
        torch.ones_like(integrals),
        atol=1.0e-5,
    )


def test_kernel_reconstruction():
    omega = torch.linspace(
        -8.0,
        8.0,
        801,
    )

    tau_fraction = torch.linspace(
        0.0,
        1.0,
        40,
    )

    beta_values = torch.tensor(
        [5.0, 10.0]
    )

    loss = AnalyticContinuationLoss(
        omega=omega,
        tau_fraction=tau_fraction,
        beta_values=beta_values,
    )

    spectrum = torch.exp(
        -(omega / 0.7) ** 2
    )[None, :]

    weights = trapezoid_weights(
        omega
    )

    spectrum = (
        spectrum
        / torch.sum(
            spectrum * weights,
            dim=1,
            keepdim=True,
        )
    )

    spectrum = spectrum.repeat(
        2,
        1,
    )

    beta = torch.tensor(
        [5.0, 10.0]
    )

    G = loss.reconstruct_green(
        spectrum,
        beta,
    )

    assert G.shape == (
        2,
        40,
    )

    assert torch.all(
        torch.isfinite(G)
    )


def test_zero_loss_for_identical_target():
    omega = torch.linspace(
        -8.0,
        8.0,
        401,
    )

    tau_fraction = torch.linspace(
        0.0,
        1.0,
        40,
    )

    loss = AnalyticContinuationLoss(
        omega=omega,
        tau_fraction=tau_fraction,
        beta_values=torch.tensor(
            [10.0]
        ),
    )

    target = torch.exp(
        -(omega / 0.8) ** 2
    )[None, :]

    weights = trapezoid_weights(
        omega
    )

    target = (
        target
        / torch.sum(
            target * weights,
            dim=1,
            keepdim=True,
        )
    )

    target = target.repeat(
        4,
        1,
    )

    beta = torch.full(
        (4,),
        10.0,
    )

    G = loss.reconstruct_green(
        target,
        beta,
    )

    values = loss(
        prediction=target,
        target=target,
        g_input=G,
        beta=beta,
    )

    assert values["loss"] < 1.0e-12
    assert values["spectrum_loss"] < 1.0e-12
    assert values["green_loss"] < 1.0e-12


def test_backward_pass():
    omega = torch.linspace(
        -8.0,
        8.0,
        401,
    )

    model = HubbardMLP(
        omega=omega,
        n_tau=40,
        hidden_dim=32,
        parameter_dim=8,
    )

    loss = AnalyticContinuationLoss(
        omega=omega,
        tau_fraction=torch.linspace(
            0.0,
            1.0,
            40,
        ),
        beta_values=torch.tensor(
            [10.0]
        ),
    )

    G = torch.randn(
        4,
        40,
    )

    conditioning = torch.tensor(
        [
            [4.0, 10.0, 1.0, 1.0e-4]
        ]
    ).repeat(
        4,
        1,
    )

    target = torch.rand(
        4,
        401,
    )

    weights = trapezoid_weights(
        omega
    )

    target = target / torch.sum(
        target * weights,
        dim=1,
        keepdim=True,
    )

    prediction = model(
        G,
        conditioning,
    )

    values = loss(
        prediction=prediction,
        target=target,
        g_input=G,
        beta=conditioning[:, 1],
    )

    values["loss"].backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert all(
        torch.all(
            torch.isfinite(gradient)
        )
        for gradient in gradients
    )
