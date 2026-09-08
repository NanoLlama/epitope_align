"""Grantham (1974) chemical-difference distances between amino acids.

The 20x20 matrix is regenerated from the published atomic property table
(composition ``c``, polarity ``p``, side-chain volume ``v``) using Grantham's
own formula rather than being transcribed as 190 hand-typed numbers:

    D(i, j) = rho * [ alpha*(c_i - c_j)^2
                    + beta *(p_i - p_j)^2
                    + gamma*(v_i - v_j)^2 ] ** 0.5

with alpha = 1.833, beta = 0.1018, gamma = 0.000399 and rho = 50.723.

Reconstructed values agree with the published matrix to within +/-2 units
(rounding in the published constants); ``tests/test_grantham.py`` pins a set of
well-known entries.

Reference: Grantham R. "Amino acid difference formula to help explain protein
evolution." Science 185:862-864 (1974).
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# amino acid -> (composition, polarity, volume)
_PROPERTIES: Dict[str, Tuple[float, float, float]] = {
    "S": (1.42, 9.2, 32.0),
    "R": (0.65, 10.5, 124.0),
    "L": (0.00, 4.9, 111.0),
    "P": (0.39, 8.0, 32.5),
    "T": (0.71, 8.6, 61.0),
    "A": (0.00, 8.1, 31.0),
    "V": (0.00, 5.9, 84.0),
    "G": (0.74, 9.0, 3.0),
    "I": (0.00, 5.2, 111.0),
    "F": (0.00, 5.2, 132.0),
    "Y": (0.20, 6.2, 136.0),
    "C": (2.75, 5.5, 55.0),
    "H": (0.58, 10.4, 96.0),
    "Q": (0.89, 10.5, 85.0),
    "N": (1.33, 11.6, 56.0),
    "K": (0.33, 11.3, 119.0),
    "D": (1.38, 13.0, 54.0),
    "E": (0.92, 12.3, 83.0),
    "M": (0.00, 5.7, 105.0),
    "W": (0.13, 5.4, 170.0),
}

_ALPHA = 1.833
_BETA = 0.1018
_GAMMA = 0.000399
_RHO = 50.723

#: Largest Grantham distance in the published matrix (Cys <-> Trp).
GRANTHAM_MAX = 215.0

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _build_matrix() -> Dict[Tuple[str, str], float]:
    matrix: Dict[Tuple[str, str], float] = {}
    for a, (ca, pa, va) in _PROPERTIES.items():
        for b, (cb, pb, vb) in _PROPERTIES.items():
            d = _RHO * math.sqrt(
                _ALPHA * (ca - cb) ** 2
                + _BETA * (pa - pb) ** 2
                + _GAMMA * (va - vb) ** 2
            )
            matrix[(a, b)] = round(d)
    return matrix


GRANTHAM: Dict[Tuple[str, str], float] = _build_matrix()


def grantham_distance(a: str, b: str) -> float:
    """Raw Grantham distance between two one-letter codes.

    Unknown/ambiguous codes (X, B, Z, U, ...) return ``float('nan')`` so callers
    can treat them as uninformative rather than as a difference of zero.
    """
    a = (a or "").upper()
    b = (b or "").upper()
    try:
        return GRANTHAM[(a, b)]
    except KeyError:
        return float("nan")


def normalized_grantham(a: str, b: str) -> float:
    """Grantham distance scaled to ``[0, 1]`` (0 = identical, 1 = C<->W)."""
    d = grantham_distance(a, b)
    if d != d:  # NaN
        return float("nan")
    return min(1.0, d / GRANTHAM_MAX)
