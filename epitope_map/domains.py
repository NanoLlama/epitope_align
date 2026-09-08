"""Structural domain decomposition from the model's own contact graph.

UniProt's feature table lists Pfam-style motifs and functional regions - "PA",
"Ligand-binding", "Mediates interaction with SH3BP4". Those are not the domain
architecture anyone would swap, and using them for that produced a recommended
construct containing one of a patch's six residues.

Real structural domains are compact, densely self-contacting units, and they are
frequently discontinuous in sequence (TfR1's protease-like domain is 121-188 plus
384-606). Spectral bisection of the C-alpha contact graph finds exactly that, from
the model alone, with no database lookup.

Preference order for what the pipeline uses:

1. ``--domains`` TSV, if the user supplies one
2. this decomposition, for any structure
3. UniProt regions - kept as annotation, never used for swap recommendations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .structure import StructureModel, distance

#: Two residues are in contact below this distance between C-alpha atoms.
CONTACT_DISTANCE = 8.0

#: Domains smaller than this are not domains.
MIN_DOMAIN_SIZE = 40

#: A split is only accepted if it cuts a small enough fraction of the contacts
#: crossing it: a compact single domain has no good cut and must not be forced.
MAX_CUT_RATIO = 0.11

#: Stop after this many domains, whatever the graph says.
MAX_DOMAINS = 8


@dataclass
class StructuralDomain:
    name: str
    ref_indices: List[int]
    source: str = "contact-graph"

    @property
    def size(self) -> int:
        return len(self.ref_indices)

    def ranges(self) -> List[Tuple[int, int]]:
        """Contiguous stretches, since a domain need not be one segment."""
        if not self.ref_indices:
            return []
        ordered = sorted(self.ref_indices)
        spans, start, previous = [], ordered[0], ordered[0]
        for index in ordered[1:]:
            if index == previous + 1:
                previous = index
                continue
            spans.append((start, previous))
            start = previous = index
        spans.append((start, previous))
        return spans


def _contact_matrix(
    points: Sequence[Optional[Tuple[float, float, float]]],
    cutoff: float = CONTACT_DISTANCE,
):
    import numpy as np

    n = len(points)
    adjacency = np.zeros((n, n), dtype=float)
    for i in range(n):
        if points[i] is None:
            continue
        for j in range(i + 1, n):
            if points[j] is None:
                continue
            if distance(points[i], points[j]) <= cutoff:
                adjacency[i, j] = adjacency[j, i] = 1.0
    # chain connectivity, so sequence neighbours are never separated by noise
    for i in range(n - 1):
        adjacency[i, i + 1] = adjacency[i + 1, i] = max(adjacency[i, i + 1], 1.0)
    return adjacency


def _fiedler_split(adjacency) -> Optional[Tuple[List[int], List[int], float]]:
    """Bisect a contact graph, returning the halves and the cut ratio."""
    import numpy as np

    degrees = adjacency.sum(axis=1)
    if (degrees <= 0).any() or adjacency.shape[0] < 4:
        return None
    with np.errstate(divide="ignore"):
        inverse_sqrt = 1.0 / np.sqrt(degrees)
    normalized = np.eye(adjacency.shape[0]) - (
        adjacency * inverse_sqrt[:, None] * inverse_sqrt[None, :]
    )
    values, vectors = np.linalg.eigh(normalized)
    order = np.argsort(values)
    if len(order) < 2:
        return None
    fiedler = vectors[:, order[1]]

    best = None
    for threshold in np.percentile(fiedler, [30, 40, 50, 60, 70]):
        left = [i for i, value in enumerate(fiedler) if value <= threshold]
        right = [i for i, value in enumerate(fiedler) if value > threshold]
        if not left or not right:
            continue
        cut = float(adjacency[np.ix_(left, right)].sum())
        total = float(adjacency.sum()) / 2.0
        ratio = cut / total if total else 1.0
        if best is None or ratio < best[2]:
            best = (left, right, ratio)
    return best


def decompose(
    structure: StructureModel,
    ref_index_of_key,
    min_size: int = MIN_DOMAIN_SIZE,
    max_cut_ratio: float = MAX_CUT_RATIO,
    max_domains: int = MAX_DOMAINS,
) -> List[StructuralDomain]:
    """Split a structure into compact, densely self-contacting units.

    Returns a single domain covering everything when no good cut exists, which
    is the honest answer for a small globular protein - forcing a split there
    would invent boundaries.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        return []

    indices: List[int] = []
    points: List[Optional[Tuple[float, float, float]]] = []
    for record in structure.residues:
        ref_index = ref_index_of_key(record.key)
        if ref_index is None:
            continue
        indices.append(ref_index)
        points.append(record.ca or record.centroid)
    if len(indices) < 2 * min_size:
        return [StructuralDomain(name="whole chain", ref_indices=list(indices))]

    adjacency = _contact_matrix(points)
    groups: List[List[int]] = [list(range(len(indices)))]
    finished: List[List[int]] = []

    while groups and len(finished) + len(groups) < max_domains:
        group = groups.pop(0)
        if len(group) < 2 * min_size:
            finished.append(group)
            continue
        sub = adjacency[np.ix_(group, group)]
        split = _fiedler_split(sub)
        if split is None:
            finished.append(group)
            continue
        left, right, ratio = split
        if (
            ratio > max_cut_ratio
            or len(left) < min_size
            or len(right) < min_size
        ):
            finished.append(group)
            continue
        groups.append([group[i] for i in left])
        groups.append([group[i] for i in right])
    finished.extend(groups)

    finished.sort(key=lambda group: min(indices[i] for i in group))
    domains: List[StructuralDomain] = []
    for number, group in enumerate(finished, start=1):
        members = sorted(indices[i] for i in group)
        name = f"D{number}" if len(finished) > 1 else "whole chain"
        domains.append(StructuralDomain(name=name, ref_indices=members))
    return domains


def describe(domain: StructuralDomain, residue_map) -> str:
    """Human-readable ranges in reference numbering."""
    parts = []
    for start, end in domain.ranges():
        first = residue_map.number_of(start) or str(start + 1)
        last = residue_map.number_of(end) or str(end + 1)
        parts.append(f"{first}-{last}")
    return "+".join(parts)
