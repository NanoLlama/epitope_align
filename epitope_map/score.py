"""Discrimination scoring, degeneracy calibration and the composite score."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .align import GAP, ResidueMap
from .data.grantham import grantham_distance, normalized_grantham
from .io_seq import Dataset

MISSING = "?"  # residue absent because the alignment ran off the end of a sequence

#: A column is called "discriminating" (for background-rate calibration and for
#: patch seeding) at or above this discrimination score.
#:
#: Note on scale: discrimination is measured in *normalized Grantham* units,
#: where 1.0 is the largest chemical difference in the matrix (Cys<->Trp). A
#: position that is perfectly conserved in binders and uniformly substituted in
#: non-binders therefore lands around 0.2-0.6 for ordinary substitutions, not
#: near 1.0 - only drastic swaps approach the top of the scale. The companion
#: ``pattern_consistency`` field is the identity-based version of the same
#: quantity and does reach 1.0 for a perfectly clean pattern.
DEFAULT_DISCRIMINATION_CUTOFF = 0.25

#: Gap-vs-residue is treated as this normalized difference: large, but flagged
#: separately because indels are harder to interpret and to mutate than a
#: point substitution.
GAP_DIFFERENCE = 0.9


@dataclass
class ColumnScore:
    """Discrimination result for one reference position."""

    ref_index: int
    column: int
    ref_aa: str
    residues: Dict[str, str] = field(default_factory=dict)
    between: float = float("nan")
    within_binders: float = float("nan")
    within_non_binders: float = float("nan")
    discrimination: float = 0.0
    pattern_consistency: float = 0.0
    max_grantham: float = 0.0
    conserved_in_binders: bool = False
    involves_gap: bool = False
    has_missing: bool = False
    low_confidence: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def is_indel(self) -> bool:
        return self.involves_gap


def _species_residue(seq: str, column: int, first: int, last: int) -> str:
    """Residue for one species at a column, distinguishing indel from missing."""
    char = seq[column]
    if char != GAP:
        return char
    return GAP if first <= column <= last else MISSING


def _span(seq: str) -> Tuple[int, int]:
    first = 0
    while first < len(seq) and seq[first] == GAP:
        first += 1
    last = len(seq) - 1
    while last >= 0 and seq[last] == GAP:
        last -= 1
    return first, last


def pair_difference(a: str, b: str) -> Optional[float]:
    """Normalized difference between two aligned residues, or ``None`` if unusable."""
    if a == MISSING or b == MISSING:
        return None
    if a == GAP and b == GAP:
        return 0.0
    if a == GAP or b == GAP:
        return GAP_DIFFERENCE
    value = normalized_grantham(a, b)
    if value != value:  # NaN: X / B / Z etc.
        return None
    return value


def _mean(values: Iterable[Optional[float]]) -> float:
    kept = [v for v in values if v is not None]
    if not kept:
        return float("nan")
    return sum(kept) / len(kept)


def identity_difference(a: str, b: str) -> Optional[float]:
    """Binary version of :func:`pair_difference`: 0 if identical, else 1."""
    if a == MISSING or b == MISSING:
        return None
    if a == GAP and b == GAP:
        return 0.0
    return 0.0 if a == b else 1.0


def _cross_mean(group_a: Sequence[str], group_b: Sequence[str]) -> float:
    return _mean(pair_difference(a, b) for a in group_a for b in group_b)


def _within_mean(group: Sequence[str]) -> float:
    if len(group) < 2:
        return 0.0
    return _mean(
        pair_difference(a, b) for a, b in itertools.combinations(group, 2)
    )


def pattern_consistency(
    residues: Dict[str, str],
    binders: Sequence[str],
    non_binders: Sequence[str],
) -> float:
    """Identity-based analogue of the discrimination score, in ``[-1, 1]``.

    1.0 means every binder carries the same residue and every non-binder carries
    a different one - the textbook clean pattern.
    """
    b = [residues[name] for name in binders]
    n = [residues[name] for name in non_binders]
    cross = _mean(identity_difference(x, y) for x in b for y in n)
    within = (
        0.0
        if len(b) < 2
        else _mean(identity_difference(x, y) for x, y in itertools.combinations(b, 2))
    )
    if cross != cross:
        return 0.0
    return cross - (0.0 if within != within else within)


def score_column(
    residues: Dict[str, str],
    binders: Sequence[str],
    non_binders: Sequence[str],
) -> Tuple[float, float, float, float]:
    """Return ``(discrimination, between, within_binders, within_non_binders)``.

    ``discrimination = between - within_binders`` per the spec: a position that
    is conserved among binders and chemically different in the non-binders
    scores near 1; a position that also varies among binders is penalised.
    """
    b = [residues[name] for name in binders]
    n = [residues[name] for name in non_binders]
    between = _cross_mean(b, n)
    within_b = _within_mean(b)
    within_n = _within_mean(n)
    if between != between:
        return 0.0, between, within_b, within_n
    penalty = 0.0 if within_b != within_b else within_b
    return between - penalty, between, within_b, within_n


def score_alignment(
    residue_map: ResidueMap,
    dataset: Dataset,
) -> List[ColumnScore]:
    """Score every reference position against the binding pattern."""
    binders = [r.name for r in dataset.binders]
    non_binders = [r.name for r in dataset.non_binders]
    spans = {
        name: _span(seq) for name, seq in residue_map.alignment.sequences.items()
    }

    scores: List[ColumnScore] = []
    for position in residue_map.positions():
        residues = {
            name: _species_residue(seq, position.column, *spans[name])
            for name, seq in residue_map.alignment.sequences.items()
        }
        discrimination, between, within_b, within_n = score_column(
            residues, binders, non_binders
        )
        scored_residues = [residues[n] for n in binders + non_binders]
        involves_gap = any(r == GAP for r in scored_residues)
        has_missing = any(r == MISSING for r in scored_residues)
        binder_residues = {residues[n] for n in binders}

        real_pairs = [
            grantham_distance(residues[b], residues[n])
            for b in binders
            for n in non_binders
            if residues[b] not in (GAP, MISSING) and residues[n] not in (GAP, MISSING)
        ]
        real_pairs = [d for d in real_pairs if d == d]

        score = ColumnScore(
            ref_index=position.ref_index,
            column=position.column,
            ref_aa=position.aa,
            residues=residues,
            between=between,
            within_binders=within_b,
            within_non_binders=within_n,
            discrimination=max(0.0, discrimination) if discrimination == discrimination else 0.0,
            pattern_consistency=pattern_consistency(residues, binders, non_binders),
            max_grantham=max(real_pairs) if real_pairs else 0.0,
            conserved_in_binders=len(binder_residues) == 1
            and GAP not in binder_residues
            and MISSING not in binder_residues,
            involves_gap=involves_gap,
            has_missing=has_missing,
            low_confidence=has_missing,
        )
        if involves_gap:
            score.notes.append(
                "indel: gap in one group and residue in the other - strong signal "
                "but harder to interpret and to mutate than a substitution"
            )
        if has_missing:
            score.notes.append(
                "a scored species has no residue here (alignment edge); scored on "
                "the remaining species only and marked low-confidence"
            )
        if residue_map.insertion_columns.get(position.ref_index):
            score.notes.append(
                f"{len(residue_map.insertion_columns[position.ref_index])} inserted "
                "alignment column(s) follow this reference residue in other species"
            )
        scores.append(score)
    return scores


# --------------------------------------------------------------------------
# degeneracy calibration
# --------------------------------------------------------------------------


@dataclass
class DegeneracyReport:
    """How much of the observed discrimination is explainable by chance."""

    n_binders: int
    n_non_binders: int
    observed_fraction: float
    background_fraction: float
    enrichment: float
    n_labelings: int
    exhaustive: bool
    clade_split: bool
    clade_split_detail: str
    cutoff: float

    @property
    def is_degenerate(self) -> bool:
        return self.clade_split or self.enrichment < 1.5


def _fraction_discriminating(
    residues_per_column: Sequence[Dict[str, str]],
    binders: Sequence[str],
    non_binders: Sequence[str],
    cutoff: float,
) -> float:
    hits = 0
    for residues in residues_per_column:
        discrimination, _, _, _ = score_column(residues, binders, non_binders)
        if discrimination >= cutoff:
            hits += 1
    return hits / len(residues_per_column) if residues_per_column else 0.0


def assess_degeneracy(
    scores: Sequence[ColumnScore],
    dataset: Dataset,
    cutoff: float = DEFAULT_DISCRIMINATION_CUTOFF,
    max_labelings: int = 200,
    seed: int = 0,
) -> DegeneracyReport:
    """Calibrate the observed signal against relabelled binding assignments.

    The background rate is the fraction of columns that look discriminating when
    the same group sizes are assigned to species at random. With two clades and
    few species this rate is high - which is the point: it tells the user how
    weak the real signal is.
    """
    binders = [r.name for r in dataset.binders]
    non_binders = [r.name for r in dataset.non_binders]
    names = binders + non_binders
    columns = [s.residues for s in scores]
    observed = _fraction_discriminating(columns, binders, non_binders, cutoff)

    observed_set = set(binders)
    rng = random.Random(seed)
    # cost per labelling grows with the square of the species count; a couple of
    # dozen relabellings already pin the background rate closely enough, so the
    # sample size is scaled down rather than letting large panels crawl
    if len(names) > 8:
        max_labelings = max(20, int(max_labelings * (8.0 / len(names)) ** 2))
    total_splits = math.comb(len(names), len(binders))
    if total_splits > 50_000:
        # too many labellings to enumerate; draw distinct ones at random
        alternatives = []
        seen = {frozenset(observed_set)}
        for _ in range(max_labelings * 20):
            if len(alternatives) >= max_labelings:
                break
            candidate = frozenset(rng.sample(names, len(binders)))
            if candidate in seen or frozenset(set(names) - candidate) in seen:
                continue
            seen.add(candidate)
            alternatives.append(set(candidate))
        exhaustive = False
    else:
        all_splits = [
            set(combo)
            for combo in itertools.combinations(names, len(binders))
            if 0 < len(combo) < len(names)
        ]
        alternatives = [
            s for s in all_splits
            if s != observed_set and set(names) - s != observed_set
        ]
        exhaustive = len(alternatives) <= max_labelings
        if not exhaustive:
            alternatives = rng.sample(alternatives, max_labelings)

    rates = [
        _fraction_discriminating(
            columns, sorted(split), sorted(set(names) - split), cutoff
        )
        for split in alternatives
    ]
    background = sum(rates) / len(rates) if rates else float("nan")
    enrichment = (
        observed / background if background and background == background and background > 0
        else float("inf") if observed > 0 else 0.0
    )
    split, detail = _clade_split_check(dataset)
    return DegeneracyReport(
        n_binders=len(binders),
        n_non_binders=len(non_binders),
        observed_fraction=observed,
        background_fraction=background,
        enrichment=enrichment,
        n_labelings=len(alternatives),
        exhaustive=exhaustive,
        clade_split=split,
        clade_split_detail=detail,
        cutoff=cutoff,
    )


def _identity_distances(dataset: Dataset, alignment_sequences: Dict[str, str]) -> Dict[Tuple[str, str], float]:
    from .align import percent_identity

    names = [r.name for r in dataset.scored]
    out: Dict[Tuple[str, str], float] = {}
    for a, b in itertools.combinations(names, 2):
        identity = percent_identity(alignment_sequences[a], alignment_sequences[b])
        out[(a, b)] = out[(b, a)] = 1.0 - identity / 100.0
    return out


def upgma_root_split(
    names: Sequence[str], distances: Dict[Tuple[str, str], float]
) -> Tuple[set, set]:
    """Cluster with UPGMA and return the two groups joined last (the root split)."""
    clusters: List[set] = [{n} for n in names]
    cluster_distance: Dict[Tuple[int, int], float] = {}
    for i, j in itertools.combinations(range(len(clusters)), 2):
        cluster_distance[(i, j)] = distances[(names[i], names[j])]

    active = {i: clusters[i] for i in range(len(clusters))}
    dist = {frozenset(k): v for k, v in cluster_distance.items()}
    next_id = len(clusters)
    while len(active) > 2:
        pair = min(
            (frozenset((i, j)) for i, j in itertools.combinations(active, 2)),
            key=lambda key: dist[key],
        )
        i, j = tuple(pair)
        merged = active[i] | active[j]
        size_i, size_j = len(active[i]), len(active[j])
        for k in list(active):
            if k in (i, j):
                continue
            dist[frozenset((next_id, k))] = (
                size_i * dist[frozenset((i, k))] + size_j * dist[frozenset((j, k))]
            ) / (size_i + size_j)
        del active[i]
        del active[j]
        active[next_id] = merged
        next_id += 1
    groups = list(active.values())
    if len(groups) == 1:  # pragma: no cover - only with a single species
        return groups[0], set()
    return groups[0], groups[1]


def _clade_split_check(dataset: Dataset) -> Tuple[bool, str]:
    """Do binders and non-binders coincide with the top-level sequence split?"""
    scored = dataset.scored
    if len(scored) < 3:
        return True, "fewer than 3 scored species: the comparison is trivially degenerate"
    return_default = (False, "binding pattern cuts across the sequence tree")
    if not hasattr(dataset, "_alignment_sequences"):
        return return_default
    sequences = getattr(dataset, "_alignment_sequences")
    names = [r.name for r in scored]
    distances = _identity_distances(dataset, sequences)
    left, right = upgma_root_split(names, distances)
    binders = {r.name for r in dataset.binders}
    non_binders = {r.name for r in dataset.non_binders}
    if (left, right) in ((binders, non_binders), (non_binders, binders)):
        return (
            True,
            "binders and non-binders correspond exactly to the top-level split of "
            f"the sequence tree ({sorted(left)} vs {sorted(right)}), so every "
            "position where the two clades diverged neutrally looks discriminating",
        )
    return return_default


# --------------------------------------------------------------------------
# composite score
# --------------------------------------------------------------------------


def exposure_weight(rsa: float, low: float = 0.05, high: float = 0.35) -> float:
    """Smooth ramp from 0 (buried) to 1 (exposed) - deliberately not a step."""
    if rsa != rsa:
        return 0.0
    if rsa <= low:
        return 0.0
    if rsa >= high:
        return 1.0
    t = (rsa - low) / (high - low)
    return t * t * (3.0 - 2.0 * t)  # smoothstep


def confidence_weight(plddt: float, low: float = 50.0, high: float = 70.0, floor: float = 0.5) -> float:
    """Down-weight low-pLDDT residues without discarding them.

    Flexible loops score low in pLDDT and are exactly where epitopes often sit,
    so the weight bottoms out at ``floor`` rather than at zero.
    """
    if plddt != plddt:
        return 1.0  # not an AlphaFold model: no confidence information to apply
    if plddt >= high:
        return 1.0
    if plddt <= low:
        return floor
    t = (plddt - low) / (high - low)
    return floor + (1.0 - floor) * t


def composite_score(discrimination: float, rsa: float, plddt: float) -> Tuple[float, float, float]:
    """Return ``(composite, exposure_weight, confidence_weight)``."""
    e = exposure_weight(rsa)
    c = confidence_weight(plddt)
    return discrimination * e * c, e, c


@dataclass
class ResidueAnalysis:
    """Everything known about one reference residue, after every stage.

    This is the row that ends up in ``residues.tsv``; the three composite factors
    are kept separate so a user can re-weight without rerunning the pipeline.
    """

    ref_index: int
    column: int
    aa: str
    ref_number: Optional[str] = None
    chain: Optional[str] = None
    species_residues: Dict[str, str] = field(default_factory=dict)

    # sequence signal
    discrimination: float = 0.0
    pattern_consistency: float = 0.0
    max_grantham: float = 0.0
    conserved_in_binders: bool = False
    involves_gap: bool = False
    has_missing: bool = False

    # structure
    modelled: bool = False
    sasa: float = float("nan")
    rsa: float = float("nan")
    plddt: float = float("nan")
    secondary_structure: str = "-"
    centroid: Optional[Tuple[float, float, float]] = None

    # weights and composite
    exposure_weight: float = 0.0
    confidence_weight: float = 1.0
    composite: float = 0.0

    # flags
    buried: bool = False
    in_ectodomain: bool = True
    mask_reasons: List[str] = field(default_factory=list)
    glycan_flags: List[str] = field(default_factory=list)
    is_sequon_asn: bool = False
    notes: List[str] = field(default_factory=list)
    patch_id: Optional[str] = None

    @property
    def masked(self) -> bool:
        return bool(self.mask_reasons)

    def eligible(self, discrimination_cutoff: float) -> bool:
        """May this residue seed a patch?"""
        return (
            not self.masked
            and self.modelled
            and self.centroid is not None
            and self.discrimination >= discrimination_cutoff
        )

    @property
    def plddt_flag(self) -> str:
        if self.plddt != self.plddt:
            return ""
        if self.plddt < 50:
            return "unreliable"
        if self.plddt < 70:
            return "low_confidence"
        return ""
