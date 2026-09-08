"""Maximum accessible surface area per residue, Tien et al. (2013).

Theoretical Gly-X-Gly values from Table 1 of:
Tien MZ, Meyer AG, Sydykova DK, Spielman SJ, Wilke CO. "Maximum allowed
solvent accessibilities of residues in proteins." PLoS ONE 8(11):e80635 (2013).

Hard-coded on purpose: the spec forbids depending on an external data file.
"""

from __future__ import annotations

from typing import Dict

MAX_ASA_TIEN: Dict[str, float] = {
    "A": 129.0,
    "R": 274.0,
    "N": 195.0,
    "D": 193.0,
    "C": 167.0,
    "E": 223.0,
    "Q": 225.0,
    "G": 104.0,
    "H": 224.0,
    "I": 197.0,
    "L": 201.0,
    "K": 236.0,
    "M": 224.0,
    "F": 240.0,
    "P": 159.0,
    "S": 155.0,
    "T": 172.0,
    "W": 285.0,
    "Y": 263.0,
    "V": 174.0,
}

#: Used for non-standard residues so they get an RSA rather than a crash.
_DEFAULT_MAX_ASA = 200.0


def max_asa(aa: str) -> float:
    """Theoretical maximum ASA (A^2) for a one-letter amino acid code."""
    return MAX_ASA_TIEN.get((aa or "").upper(), _DEFAULT_MAX_ASA)


def relative_sasa(aa: str, sasa: float) -> float:
    """Relative solvent accessibility, clamped to ``[0, 1.5]``.

    Values slightly above 1.0 occur for terminal or unusually extended residues;
    they are kept (not clipped to 1.0) up to a sane ceiling so the ramp in
    :mod:`epitope_map.score` behaves monotonically.
    """
    if sasa is None or sasa != sasa:
        return float("nan")
    return max(0.0, min(1.5, sasa / max_asa(aa)))
