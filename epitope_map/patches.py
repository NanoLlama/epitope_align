"""Spatial clustering of discriminating surface residues into candidate patches.

This is the stage that does the real narrowing: sequence alone leaves 50-150
"discriminating" positions in a two-clade comparison, and only their spatial
arrangement separates a plausible epitope from scattered neutral drift.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

#: Maximum diameter a candidate merged surface may reach while it is being
#: grown. Exposed as --footprint-diameter.
FOOTPRINT_DIAMETER = 30.0

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
    membership: str = ""
    confidence_penalty: float = 1.0
    penalty_reasons: List[str] = field(default_factory=list)
    neighbours: List[Tuple[str, float]] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    # ---- metrics
    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def raw_total_score(self) -> float:
        return sum(m.composite for m in self.members)

    @property
    def total_score(self) -> float:
        """Summed composite, after any confidence penalty.

        A penalty here is not cosmetic: a patch sitting on an unmodelled
        oligomer interface, or one that is entirely glycan-proximal, is a weaker
        hypothesis than its residue scores suggest and should rank as one.
        """
        return self.raw_total_score * self.confidence_penalty

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


def patch_identifier(members: Sequence[ResidueAnalysis]) -> str:
    """A patch's name, derived from its contents rather than its rank.

    Letters assigned by score order are reassigned whenever anything shifts, so
    "patch PD" in yesterday's notes silently means a different set of residues
    today - which matters for a tool whose whole purpose is re-running under
    changed parameters. Naming a patch after its lowest member residue is
    stable, unique (a residue belongs to one patch) and readable at a glance.
    """
    if not members:
        return "p:empty"
    lowest = min(members, key=lambda m: m.ref_index)
    return f"p:{lowest.ref_number or lowest.ref_index + 1}"


def membership_hash(members: Sequence[ResidueAnalysis]) -> str:
    """Short digest of the exact membership, for spotting drift between runs."""
    key = ",".join(
        sorted(str(m.ref_number or m.ref_index + 1) for m in members)
    )
    return hashlib.sha1(key.encode()).hexdigest()[:8]


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
        patch = Patch(patch_id=patch_identifier(group), members=group)
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
    footprint_diameter: float = FOOTPRINT_DIAMETER,
    max_groups: int = 6,
) -> List[Dict[str, object]]:
    """Candidate surfaces an antibody could actually cover, grown by diameter.

    Transitive closure at any link distance collapses to the whole protein on a
    real target - every patch is within 25 A of some other patch - and returns
    one useless group labelled "too large". Growing by *diameter* instead asks
    the question that matters: is there a set of patches whose members all fit
    inside one footprint? A merge that would breach the diameter is rejected
    rather than accepted and then apologised for.

    Groups are seeded from each patch in turn, so they overlap; that is expected
    and they are reported as alternative groupings.
    """
    everything = [p for p in list(patches) + list(singletons) if p.members]
    if not everything:
        return []

    gaps: Dict[Tuple[str, str], float] = {}
    for a in everything:
        for b in everything:
            if a.patch_id != b.patch_id:
                gaps[(a.patch_id, b.patch_id)] = a.min_distance_to(b)

    seen: Dict[frozenset, Dict[str, object]] = {}
    grouped: set = set()
    for seed in sorted(everything, key=lambda p: -p.total_score):
        group = [seed]
        members = list(seed.members)
        while True:
            best, best_key = None, None
            for candidate in everything:
                # compare by identifier: Patch is a dataclass, so `in` would run
                # a deep field comparison and could match the wrong object
                if any(candidate.patch_id == p.patch_id for p in group):
                    continue
                gap = min(
                    gaps[(candidate.patch_id, member.patch_id)] for member in group
                )
                if gap > footprint_diameter:
                    continue  # not a neighbour of this group at all
                combined = members + candidate.members
                diameter = _diameter(combined)
                if diameter > footprint_diameter:
                    continue  # the merge that would breach the footprint is refused
                # take the partner that keeps the group tightest, not the
                # highest-scoring one: a 21 A partner should never be chosen
                # over a 14 A one that also fits
                key = (round(diameter, 3), round(gap, 3), -candidate.total_score)
                if best_key is None or key < best_key:
                    best, best_key = candidate, key
            if best is None:
                break
            group.append(best)
            members = members + best.members

        if len(group) < 2:
            continue
        grouped.update(p.patch_id for p in group)
        key = frozenset(p.patch_id for p in group)
        if key in seen:
            continue

        unique: Dict[int, ResidueAnalysis] = {m.ref_index: m for m in members}
        residues = sorted(unique.values(), key=lambda m: m.ref_index)
        # union, not a sum of independently computed patch areas: overlapping
        # neighbourhoods double-count, which inflated the old figure
        area = sum(m.sasa for m in residues if m.sasa == m.sasa)
        seen[key] = {
            "patches": sorted(p.patch_id for p in group),
            "grown_from": seed.patch_id,
            "n_residues": len(residues),
            "verdict": _surface_verdict(residues, _diameter(residues), area),
            "residues": [m.ref_number or str(m.ref_index + 1) for m in residues],
            "total_score": sum(m.composite for m in residues),
            "accessible_area_A2": area,
            "spread_A": _diameter(residues),
            "max_gap_A": max(
                (
                    gaps[(a.patch_id, b.patch_id)]
                    for a in group
                    for b in group
                    if a.patch_id != b.patch_id
                ),
                default=0.0,
            ),
        }

    out = sorted(seen.values(), key=lambda entry: -float(entry["total_score"]))
    out = out[:max_groups]

    # nothing may vanish: a patch that joins no group is reported as its own
    # candidate surface, with why. Silently omitting it hid the top-ranked patch.
    for patch in sorted(everything, key=lambda p: -p.total_score):
        if patch.patch_id in grouped or any(
            patch.patch_id in entry["patches"] for entry in out
        ):
            continue
        nearest = min(
            (
                (gaps[(patch.patch_id, other.patch_id)], other.patch_id)
                for other in everything
                if other.patch_id != patch.patch_id
            ),
            default=(float("inf"), ""),
        )
        residues = sorted(patch.members, key=lambda m: m.ref_index)
        area = sum(m.sasa for m in residues if m.sasa == m.sasa)
        reason = (
            f"nearest patch {nearest[1]} is {nearest[0]:.0f} A away and merging "
            f"would exceed the {footprint_diameter:.0f} A footprint"
            if nearest[0] <= footprint_diameter
            else f"nearest patch {nearest[1]} is {nearest[0]:.0f} A away"
            if nearest[1]
            else "no other patch to group with"
        )
        out.append(
            {
                "patches": [patch.patch_id],
                "grown_from": patch.patch_id,
                "n_residues": len(residues),
                "verdict": f"stands alone - {reason}",
                "residues": [m.ref_number or str(m.ref_index + 1) for m in residues],
                "total_score": patch.total_score,
                "accessible_area_A2": area,
                "spread_A": _diameter(residues),
                "max_gap_A": 0.0,
            }
        )
    return out


def _diameter(members: Sequence[ResidueAnalysis]) -> float:
    points = [m.centroid for m in members if m.centroid is not None]
    if len(points) < 2:
        return 0.0
    return max(distance(a, b) for a, b in itertools.combinations(points, 2))


def _surface_verdict(
    residues: Sequence[ResidueAnalysis], spread: float, area: float
) -> str:
    if spread > EPITOPE_MAX_SPAN or area > 1.5 * TYPICAL_EPITOPE_BSA[1]:
        return (
            "too large for one antibody footprint - treat as neighbouring "
            "surfaces, not one epitope"
        )
    if len(residues) > TYPICAL_EPITOPE_RESIDUES[1]:
        return "larger than a typical epitope; the true footprint is a subset"
    return "plausible single epitope"


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

#: An indel promotes itself only if its geometry says it reshapes the surface;
#: a gap inside a helix is an alignment artifact and stays in the leftovers.
PROMOTION_INDEL_SCORE = 0.35


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
        if member.involves_gap and (
            member.indel_score != member.indel_score
            or member.indel_score >= PROMOTION_INDEL_SCORE
        ):
            reasons.append(
                f"indel (indel_score {member.indel_score:.2f}) - reshapes the "
                "local surface"
                if member.indel_score == member.indel_score
                else "indel - reshapes the local surface"
            )
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


#: How much a patch's score is discounted when it sits on an interface the
#: structure does not model, or when it is entirely glycan-proximal.
OLIGOMER_PENALTY = 0.6
GLYCAN_PENALTY = 0.6


def apply_confidence_penalties(
    patches: Sequence[Patch],
    oligomer_unmodelled: bool,
    interface_regions: Sequence[Tuple[int, int]] = (),
) -> None:
    """Let the warnings reach the ranking instead of only the prose.

    A patch on an oligomer interface the structure does not model, and a patch
    every member of which is glycan-proximal, are both weaker hypotheses than
    their residue scores suggest. Saying so in a warning while ranking them
    alongside everything else leaves the reader to apply the discount by hand.
    """
    for patch in patches:
        penalty = 1.0
        if patch.members and all(m.glycan_flags for m in patch.members):
            penalty *= GLYCAN_PENALTY
            patch.penalty_reasons.append(
                "every member is within reach of a differential glycosylation "
                "sequon, so this is a glycan hypothesis rather than a surface "
                f"one - score discounted x{GLYCAN_PENALTY}"
            )
        if oligomer_unmodelled and interface_regions:
            inside = [
                m
                for m in patch.members
                if any(
                    start <= m.ref_index + 1 <= end for start, end in interface_regions
                )
            ]
            if inside:
                penalty *= OLIGOMER_PENALTY
                patch.penalty_reasons.append(
                    f"{len(inside)} member(s) lie in an annotated "
                    "oligomerisation/interface region that this monomer "
                    "structure does not model, so their exposure is overstated "
                    f"- score discounted x{OLIGOMER_PENALTY}"
                )
        patch.confidence_penalty = penalty
        patch.flags.extend(patch.penalty_reasons)


def baseline_counts(
    residues: Sequence[ResidueAnalysis], discrimination_cutoff: float
) -> Dict[str, int]:
    """Candidate-set sizes at each narrowing step, for the enrichment table."""
    total = len(residues)
    # "within the analysed range" has to mean reachable, or the row contradicts
    # the topology exclusion reported directly above it
    in_range = [r for r in residues if r.accessible]
    discriminating = [r for r in in_range if r.discrimination >= discrimination_cutoff]
    exposed = [r for r in discriminating if not r.buried and r.modelled]
    unmasked = [r for r in exposed if not r.masked]
    return {
        "all_reference_residues": total,
        "reachable": len(in_range),
        "discriminating": len(discriminating),
        "discriminating_and_exposed": len(exposed),
        "after_context_masking": len(unmasked),
    }
