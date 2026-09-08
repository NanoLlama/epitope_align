"""Spatial clustering of discriminating surface residues into candidate patches.

This is the stage that does the real narrowing: sequence alone leaves 50-150
"discriminating" positions in a two-clade comparison, and only their spatial
arrangement separates a plausible epitope from scattered neutral drift.
"""

from __future__ import annotations

import itertools
import math
import string
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .score import ResidueAnalysis
from .structure import distance

DEFAULT_PATCH_RADIUS = 12.0
DEFAULT_MIN_PATCH_SIZE = 2

#: A conformational epitope is typically ~15-22 residues burying ~600-900 A^2,
#: with 4-6 energetic hot spots. Patches far outside this envelope are flagged.
TYPICAL_EPITOPE_RESIDUES = (15, 22)
TYPICAL_EPITOPE_BSA = (600.0, 900.0)
MAX_PLAUSIBLE_SPREAD = 25.0

#: An antibody paratope covers roughly a 20-25 A disc, so a cluster that spans
#: much more than this cannot be contacted as a single epitope. Compactness
#: decays linearly from 1.0 at COMPACT_SPAN to COMPACTNESS_FLOOR at DIFFUSE_SPAN
#: and feeds the size-normalised ranking.
COMPACT_SPAN = 20.0
DIFFUSE_SPAN = 35.0
COMPACTNESS_FLOOR = 0.4

#: Two patches whose nearest members are within this distance could be contacted
#: by one antibody, so they are reported as a candidate merged surface rather
#: than as independent hypotheses.
EPITOPE_SCALE = 25.0

#: A 15-22 residue epitope spans roughly 25-30 A, so a merged surface wider than
#: this cannot be covered by one paratope however close its fragments are.
EPITOPE_MAX_SPAN = 30.0

#: Radii used by --radius-sweep when the user does not name their own.
DEFAULT_RADIUS_SWEEP = (10.0, 12.0, 14.0, 16.0, 18.0)


@dataclass
class Patch:
    """A cluster of exposed, discriminating reference residues.

    ``members`` are the discriminating residues that define and score the patch.
    ``context`` are conserved exposed residues sitting inside the same surface:
    a real epitope contains conserved residues, and leaving them out understates
    how large the footprint is and what a chimera would have to cover. They do
    not contribute to the score.
    """

    patch_id: str
    members: List[ResidueAnalysis] = field(default_factory=list)
    context: List[ResidueAnalysis] = field(default_factory=list)
    rank_raw: int = 0
    rank_normalized: int = 0
    rank_combined: int = 0
    promoted_reason: str = ""
    neighbours: List[Tuple[str, float]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    # ---- metrics
    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def total_score(self) -> float:
        return sum(m.composite for m in self.members)

    @property
    def mean_score(self) -> float:
        return self.total_score / self.size if self.size else 0.0

    @property
    def compactness(self) -> float:
        """1.0 for a patch an antibody could cover, decaying as it spreads out."""
        spread = self.spread
        if spread <= COMPACT_SPAN:
            return 1.0
        if spread >= DIFFUSE_SPAN:
            return COMPACTNESS_FLOOR
        t = (spread - COMPACT_SPAN) / (DIFFUSE_SPAN - COMPACT_SPAN)
        return 1.0 - (1.0 - COMPACTNESS_FLOOR) * t

    @property
    def normalized_score(self) -> float:
        """Total score damped by sqrt(size) and by how far the patch spreads.

        Size normalisation alone is not enough: a chain of weakly discriminating
        residues running across the whole surface accumulates both a high total
        and a large residue count. Folding in compactness keeps a tight,
        chemically drastic patch ahead of a diffuse one, which is what the
        ranking is for. The raw ranking is reported unchanged alongside it.
        """
        if not self.size:
            return 0.0
        return (self.total_score / math.sqrt(self.size)) * self.compactness

    @property
    def n_discriminating(self) -> int:
        return sum(1 for m in self.members if m.discrimination > 0)

    @property
    def n_total_surface(self) -> int:
        """Discriminating members plus the conserved surface around them."""
        return len(self.members) + len(self.context)

    @property
    def surface_area_with_context(self) -> float:
        return self.surface_area + sum(
            r.sasa for r in self.context if r.sasa == r.sasa
        )

    def min_distance_to(self, other: "Patch") -> float:
        """Closest approach between any two members - not centroid separation.

        Centroid distance overstates the separation of elongated patches, which
        is how one apical surface came back as four separate hypotheses.
        """
        best = float("inf")
        for a in self.members:
            for b in other.members:
                if a.centroid is None or b.centroid is None:
                    continue
                best = min(best, distance(a.centroid, b.centroid))
        return best

    @property
    def max_grantham(self) -> float:
        return max((m.max_grantham for m in self.members), default=0.0)

    @property
    def mean_rsa(self) -> float:
        values = [m.rsa for m in self.members if m.rsa == m.rsa]
        return sum(values) / len(values) if values else float("nan")

    @property
    def surface_area(self) -> float:
        """Approximate solvent-accessible area contributed by the members."""
        return sum(m.sasa for m in self.members if m.sasa == m.sasa)

    @property
    def centroid(self) -> Tuple[float, float, float]:
        points = [m.centroid for m in self.members if m.centroid is not None]
        if not points:
            return (float("nan"),) * 3
        n = len(points)
        return (
            sum(p[0] for p in points) / n,
            sum(p[1] for p in points) / n,
            sum(p[2] for p in points) / n,
        )

    @property
    def spread(self) -> float:
        points = [m.centroid for m in self.members if m.centroid is not None]
        if len(points) < 2:
            return 0.0
        return max(distance(a, b) for a, b in itertools.combinations(points, 2))

    @property
    def n_indels(self) -> int:
        return sum(1 for m in self.members if m.involves_gap)

    @property
    def n_glycan_flagged(self) -> int:
        return sum(1 for m in self.members if m.glycan_flags)

    @property
    def residue_labels(self) -> List[str]:
        return [f"{m.aa}{m.ref_number or f'idx{m.ref_index + 1}'}" for m in self.members]

    def rationale(self) -> str:
        parts = [
            f"{self.size} discriminating surface residue(s) "
            f"({', '.join(self.residue_labels)})",
            f"{len(self.context)} conserved surface residue(s) alongside",
            f"mean RSA {self.mean_rsa:.2f}",
            f"max Grantham {self.max_grantham:.0f}",
            f"spread {self.spread:.1f} A",
        ]
        if self.n_indels:
            parts.append(f"{self.n_indels} indel position(s)")
        if self.n_glycan_flagged:
            parts.append(f"{self.n_glycan_flagged} residue(s) near a differential sequon")
        if self.flags:
            parts.append("; ".join(self.flags))
        return "; ".join(parts)


def _patch_ids() -> Iterable[str]:
    letters = string.ascii_uppercase
    for size in range(1, 3):
        for combo in itertools.product(letters, repeat=size):
            yield "P" + "".join(combo)


def _connected_components(
    residues: Sequence[ResidueAnalysis], radius: float
) -> List[List[ResidueAnalysis]]:
    n = len(residues)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i, j in itertools.combinations(range(n), 2):
        a, b = residues[i].centroid, residues[j].centroid
        if a is None or b is None:
            continue
        if distance(a, b) <= radius:
            union(i, j)

    groups: Dict[int, List[ResidueAnalysis]] = {}
    for i, residue in enumerate(residues):
        groups.setdefault(find(i), []).append(residue)
    return list(groups.values())


def _dbscan_components(
    residues: Sequence[ResidueAnalysis], radius: float, min_samples: int
) -> List[List[ResidueAnalysis]]:
    """Textbook DBSCAN with ``eps = radius``.

    Unlike connected components, border points do not extend a cluster, so a
    single-residue bridge cannot chain two surfaces together. Noise points are
    returned as their own singleton clusters rather than being discarded.
    """
    points = [r.centroid for r in residues]
    if len(points) < 2:
        return [list(residues)] if residues else []
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        neighbours = tree.query_ball_point(points, r=radius)
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        neighbours = [
            [j for j in range(len(points)) if distance(points[i], points[j]) <= radius]
            for i in range(len(points))
        ]

    min_samples = max(2, min_samples)
    labels: Dict[int, int] = {}
    cluster = 0
    for i in range(len(points)):
        if i in labels:
            continue
        if len(neighbours[i]) < min_samples:
            continue  # not a core point; may still be claimed as a border point
        cluster += 1
        labels[i] = cluster
        queue = list(neighbours[i])
        while queue:
            j = queue.pop()
            if j in labels:
                continue
            labels[j] = cluster
            if len(neighbours[j]) >= min_samples:
                queue.extend(neighbours[j])

    groups: Dict[int, List[ResidueAnalysis]] = {}
    noise = 0
    for i, residue in enumerate(residues):
        if i in labels:
            groups.setdefault(labels[i], []).append(residue)
        else:
            noise -= 1
            groups[noise] = [residue]
    return list(groups.values())


def find_patches(
    residues: Sequence[ResidueAnalysis],
    discrimination_cutoff: float,
    radius: float = DEFAULT_PATCH_RADIUS,
    min_size: int = DEFAULT_MIN_PATCH_SIZE,
    method: str = "graph",
    rsa_cutoff: float = 0.20,
    assign_ids: bool = True,
) -> Tuple[List[Patch], List[Patch]]:
    """Cluster eligible residues into patches.

    Returns ``(patches, singletons)``: clusters below ``min_size`` are not
    discarded, they are returned separately for a low-priority report section.
    Conserved exposed residues within ``radius`` of a member are attached to each
    patch as context - they do not score, but they show the real extent of the
    surface a chimera would have to cover.
    """
    seeds = [r for r in residues if r.eligible(discrimination_cutoff)]
    if not seeds:
        return [], []

    if method == "dbscan":
        components = _dbscan_components(seeds, radius, min_size)
    else:
        components = _connected_components(seeds, radius)

    components.sort(key=lambda group: -sum(m.composite for m in group))
    ids = _patch_ids()
    seed_indices = {r.ref_index for r in seeds}
    context_pool = [
        r
        for r in residues
        if r.ref_index not in seed_indices and r.surface_context(rsa_cutoff)
    ]

    patches: List[Patch] = []
    singletons: List[Patch] = []
    for group in components:
        group.sort(key=lambda r: r.ref_index)
        patch = Patch(patch_id=next(ids), members=group)
        patch.context = _nearby_context(group, context_pool, radius)
        _annotate(patch)
        (patches if patch.size >= min_size else singletons).append(patch)

    _rank(patches)
    _rank(singletons)
    _link_neighbours(patches, singletons)
    if assign_ids:
        for patch in patches + singletons:
            for member in patch.members:
                member.patch_id = patch.patch_id
    return patches, singletons


def _nearby_context(
    members: Sequence[ResidueAnalysis],
    pool: Sequence[ResidueAnalysis],
    radius: float,
) -> List[ResidueAnalysis]:
    out = []
    for candidate in pool:
        if candidate.centroid is None:
            continue
        if any(
            m.centroid is not None and distance(candidate.centroid, m.centroid) <= radius
            for m in members
        ):
            out.append(candidate)
    return sorted(out, key=lambda r: r.ref_index)


def _link_neighbours(
    patches: Sequence[Patch],
    singletons: Sequence[Patch],
    scale: float = EPITOPE_SCALE,
) -> None:
    """Record which patches are close enough to be one epitope."""
    everything = list(patches) + list(singletons)
    for a in everything:
        for b in everything:
            if a is b:
                continue
            gap = a.min_distance_to(b)
            if gap <= scale:
                a.neighbours.append((b.patch_id, gap))
        a.neighbours.sort(key=lambda pair: pair[1])


def merged_surfaces(
    patches: Sequence[Patch],
    singletons: Sequence[Patch] = (),
    scale: float = EPITOPE_SCALE,
) -> List[Dict[str, object]]:
    """Group patches whose nearest members are within one antibody footprint.

    A 15-22 residue epitope spans roughly 25-30 A, so patches this close are one
    candidate surface, and it is their combined area that should be compared
    against the 600-900 A^2 an antibody buries - not each fragment's.
    """
    everything = list(patches) + list(singletons)
    if not everything:
        return []
    index = {patch.patch_id: i for i, patch in enumerate(everything)}
    parent = list(range(len(everything)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for patch in everything:
        for other_id, gap in patch.neighbours:
            if gap <= scale and other_id in index:
                a, b = find(index[patch.patch_id]), find(index[other_id])
                if a != b:
                    parent[b] = a

    groups: Dict[int, List[Patch]] = {}
    for i, patch in enumerate(everything):
        groups.setdefault(find(i), []).append(patch)

    out: List[Dict[str, object]] = []
    for group in groups.values():
        if len(group) < 2:
            continue
        members = [m for patch in group for m in patch.members]
        area = sum(m.sasa for m in members if m.sasa == m.sasa)
        spread = 0.0
        for a in members:
            for b in members:
                if a.centroid and b.centroid:
                    spread = max(spread, distance(a.centroid, b.centroid))
        members.sort(key=lambda m: m.ref_index)
        verdict = "plausible single epitope"
        if spread > EPITOPE_MAX_SPAN or area > 1.5 * TYPICAL_EPITOPE_BSA[1]:
            verdict = (
                "too large for one antibody footprint - treat as neighbouring "
                "surfaces, not one epitope"
            )
        elif len(members) > TYPICAL_EPITOPE_RESIDUES[1]:
            verdict = "larger than a typical epitope; the true footprint is a subset"
        out.append(
            {
                "patches": sorted(p.patch_id for p in group),
                "n_residues": len(members),
                "verdict": verdict,
                "residues": [m.ref_number or str(m.ref_index + 1) for m in members],
                "total_score": sum(m.composite for m in members),
                "accessible_area_A2": area,
                "spread_A": spread,
                "max_gap_A": max(
                    (gap for p in group for pid, gap in p.neighbours
                     if pid in {q.patch_id for q in group}),
                    default=0.0,
                ),
            }
        )
    out.sort(key=lambda entry: -float(entry["total_score"]))
    return out


def radius_sweep(
    residues: Sequence[ResidueAnalysis],
    discrimination_cutoff: float,
    radii: Sequence[float] = DEFAULT_RADIUS_SWEEP,
    min_size: int = DEFAULT_MIN_PATCH_SIZE,
    method: str = "graph",
    rsa_cutoff: float = 0.20,
) -> List[Dict[str, object]]:
    """Re-cluster at several radii and record what merges when.

    Patches that merge at a small radius are one surface; patches still separate
    at 18 A are genuinely distinct hypotheses. The clustering radius is the most
    consequential free parameter in the whole pipeline and this is the only
    honest way to show its effect.
    """
    saved = {r.ref_index: r.patch_id for r in residues}
    rows: List[Dict[str, object]] = []
    try:
        for radius in radii:
            patches, singletons = find_patches(
                residues,
                discrimination_cutoff=discrimination_cutoff,
                radius=radius,
                min_size=min_size,
                method=method,
                rsa_cutoff=rsa_cutoff,
                assign_ids=False,
            )
            groups = [
                sorted(m.ref_number or str(m.ref_index + 1) for m in patch.members)
                for patch in patches + list(singletons)
            ]
            groups.sort(key=lambda g: (-len(g), g))
            rows.append(
                {
                    "radius_A": radius,
                    "n_patches": len(patches),
                    "n_singletons": len(singletons),
                    "largest_patch": max((len(g) for g in groups), default=0),
                    "groupings": " | ".join(",".join(g) for g in groups),
                }
            )
    finally:
        for residue in residues:
            residue.patch_id = saved.get(residue.ref_index)
    return rows


def _annotate(patch: Patch) -> None:
    if patch.spread > MAX_PLAUSIBLE_SPREAD:
        patch.flags.append(
            f"spread {patch.spread:.0f} A exceeds {MAX_PLAUSIBLE_SPREAD:.0f} A - "
            "probably not a single epitope; consider splitting it or lowering "
            "--patch-radius"
        )
    low, high = TYPICAL_EPITOPE_RESIDUES
    if patch.size > high:
        patch.flags.append(
            f"{patch.size} residues is larger than a typical epitope ({low}-{high} "
            "residues); it may be merging neighbouring surfaces"
        )
    area_low, area_high = TYPICAL_EPITOPE_BSA
    if patch.surface_area and patch.surface_area < 0.4 * area_low:
        patch.flags.append(
            f"accessible area {patch.surface_area:.0f} A^2 is small next to the "
            f"{area_low:.0f}-{area_high:.0f} A^2 an antibody typically buries; a real "
            "epitope here would extend into conserved residues too"
        )
    if all(m.involves_gap for m in patch.members):
        patch.flags.append("every member is an indel position - interpret with care")

    shaky = [
        m
        for m in patch.members
        if (
            m.alignment_confidence == m.alignment_confidence
            and m.alignment_confidence < 0.7
        )
        or m.low_identity_window
    ]
    if shaky:
        patch.flags.append(
            f"{len(shaky)} of {patch.size} member(s) sit where the alignment is "
            "ambiguous ("
            + ", ".join(m.ref_number or str(m.ref_index + 1) for m in shaky[:6])
            + "): the region is a real candidate but the specific residue "
            "equivalences are not, so swap the segment rather than ordering point "
            "mutants there"
        )


def _rank(patches: List[Patch]) -> None:
    """Assign the raw, normalised and combined ranks and order by the last."""
    for rank, patch in enumerate(sorted(patches, key=lambda p: -p.total_score), start=1):
        patch.rank_raw = rank
    for rank, patch in enumerate(
        sorted(patches, key=lambda p: -p.normalized_score), start=1
    ):
        patch.rank_normalized = rank
    # presentation order: best of the two rankings, so neither a diffuse
    # high-total patch nor a small high-density one is hidden from the user
    patches.sort(
        key=lambda p: (min(p.rank_raw, p.rank_normalized), p.rank_normalized, p.rank_raw)
    )
    for rank, patch in enumerate(patches, start=1):
        patch.rank_combined = rank


#: A singleton scoring at or above this is worth surfacing on its own.
PROMOTION_SCORE = 0.5


def promote_singletons(
    singletons: Sequence[Patch],
    patches: Sequence[Patch],
    score_threshold: float = PROMOTION_SCORE,
    scale: float = EPITOPE_SCALE,
) -> Tuple[List[Patch], List[Patch]]:
    """Split isolated residues into high- and low-priority sets.

    Being alone is not evidence of being unimportant: an isolated residue is
    filed as a singleton only because no *other above-cutoff* residue sits
    within the clustering radius. A surface-loop insertion reshapes local
    geometry and is a more plausible way to abolish binding than most single
    substitutions, so it should not be buried under a dump of leftovers.
    """
    high: List[Patch] = []
    low: List[Patch] = []
    ranked_ids = {p.patch_id for p in patches}
    for patch in singletons:
        reasons: List[str] = []
        member = patch.members[0]
        if member.composite >= score_threshold:
            reasons.append(f"composite {member.composite:.2f} above {score_threshold}")
        if member.involves_gap:
            reasons.append("indel - reshapes the local surface")
        near = [
            (pid, gap) for pid, gap in patch.neighbours if pid in ranked_ids and gap <= scale
        ]
        if near:
            reasons.append(
                f"within {near[0][1]:.0f} A of {near[0][0]}, so plausibly part of "
                "that surface"
            )
        if reasons:
            patch.promoted_reason = "; ".join(reasons)
            high.append(patch)
        else:
            low.append(patch)
    high.sort(key=lambda p: -p.members[0].composite)
    return high, low


def baseline_counts(
    residues: Sequence[ResidueAnalysis], discrimination_cutoff: float
) -> Dict[str, int]:
    """Candidate-set sizes at each narrowing step, for the enrichment table."""
    total = len(residues)
    in_range = [r for r in residues if r.in_ectodomain]
    discriminating = [r for r in in_range if r.discrimination >= discrimination_cutoff]
    exposed = [r for r in discriminating if not r.buried and r.modelled]
    unmasked = [r for r in exposed if not r.masked]
    return {
        "all_reference_residues": total,
        "in_ectodomain": len(in_range),
        "discriminating": len(discriminating),
        "discriminating_and_exposed": len(exposed),
        "after_context_masking": len(unmasked),
    }
