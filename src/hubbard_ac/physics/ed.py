import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh


def full_eigensystem(matrix) -> tuple[np.ndarray, np.ndarray]:
    if issparse(matrix):
        matrix = matrix.toarray()

    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(matrix, dtype=np.float64)
    )

    order = np.argsort(eigenvalues)

    return eigenvalues[order], eigenvectors[:, order]


def lowest_eigenpairs(
    matrix,
    k: int = 1,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    n = matrix.shape[0]

    if k < 1:
        raise ValueError("k must be positive")

    if k >= n:
        return full_eigensystem(matrix)

    eigenvalues, eigenvectors = eigsh(
        matrix,
        k=k,
        which="SA",
        tol=tol,
    )

    order = np.argsort(eigenvalues)

    return eigenvalues[order], eigenvectors[:, order]


def ground_state(
    matrix,
    tol: float = 1e-12,
) -> tuple[float, np.ndarray]:
    eigenvalues, eigenvectors = lowest_eigenpairs(
        matrix,
        k=1,
        tol=tol,
    )

    return float(eigenvalues[0]), eigenvectors[:, 0]
