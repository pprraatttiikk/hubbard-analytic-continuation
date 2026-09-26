import numpy as np
import pytest

from hubbard_ac.data.noise import gaussian_noise


def test_zero_noise_returns_same_signal():
    signal = np.linspace(-1.0, 1.0, 100)

    rng = np.random.default_rng(1)

    noisy = gaussian_noise(
        signal,
        sigma=0.0,
        rng=rng,
    )

    assert np.array_equal(noisy, signal)
    assert noisy is not signal


def test_noise_is_reproducible():
    signal = np.zeros(100)

    first = gaussian_noise(
        signal,
        sigma=0.1,
        rng=np.random.default_rng(123),
    )

    second = gaussian_noise(
        signal,
        sigma=0.1,
        rng=np.random.default_rng(123),
    )

    assert np.array_equal(first, second)


def test_noise_standard_deviation():
    signal = np.zeros(200000)

    sigma = 0.02

    noisy = gaussian_noise(
        signal,
        sigma=sigma,
        rng=np.random.default_rng(123),
    )

    assert np.isclose(
        np.std(noisy),
        sigma,
        rtol=0.02,
    )


def test_negative_sigma_raises():
    with pytest.raises(ValueError):
        gaussian_noise(
            np.zeros(10),
            sigma=-0.1,
            rng=np.random.default_rng(1),
        )
