"""Experiment suggestions: chimera boundaries and reciprocal point mutants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .align import GAP, ResidueMap
from .data.grantham import grantham_distance
from .io_seq import Dataset
from .patches import Patch
from .score import MISSING, ResidueAnalysis
from .structure import StructureModel

#: DSSP codes that are safe to cut through (coil, bend, turn, 3-10 exposed ends).
_CUTTABLE_SS = set("-STC ")

#: How far a boundary may be pushed outward to escape a helix or strand.
_MAX_BOUNDARY_SHIFT = 12

#: Runs of patch members closer together than this are swapped as one segment;
#: further apart, they are proposed as separate swaps. A conformational epitope
#: is usually discontinuous, so one start-end range spanning the whole patch
#: would swap most of the domain and prove nothing.
_MERGE_GAP = 8

#: Residues of flanking sequence included on each side of a segment.
_FLANK = 2


@dataclass
class ChimeraSegment:
    """One contiguous stretch to swap."""

    start_ref_index: int
    end_ref_index: int
    start_number: Optional[str]
    end_number: Optional[str]
    ss_respected: bool

    @property
    def length(self) -> int:
        return self.end_ref_index - self.start_ref_index + 1

    @property
    def label(self) -> str:
        start = self.start_number or str(self.start_ref_index + 1)
        end = self.end_number or str(self.end_ref_index + 1)
        return f"{start}-{end}"


@dataclass
class ChimeraSuggestion:
    patch_id: str
    segments: List[ChimeraSegment] = field(default_factory=list)
    other_patches_included: List[str] = field(default_factory=list)
    n_discriminating_included: int = 0
    note: str = ""

    @property
    def start_ref_index(self) -> int:
        return min(s.start_ref_index for s in self.segments)

    @property
    def end_ref_index(self) -> int:
        return max(s.end_ref_index for s in self.segments)

    @property
    def start_number(self) -> Optional[str]:
        return self.segments[0].start_number if self.segments else None

    @property
    def end_number(self) -> Optional[str]:
        return self.segments[-1].end_number if self.segments else None

    @property
    def length(self) -> int:
        """Total residues swapped, not the span from first to last."""
        return sum(s.length for s in self.segments)

    @property
    def ss_respected(self) -> bool:
        return all(s.ss_respected for s in self.segments)

    @property
    def label(self) -> str:
        return " + ".join(s.label for s in self.segments)


@dataclass
class MutantSuggestion:
    patch_id: str
    direction: str  # loss_of_binding | gain_of_binding
    background: str
    position: Optional[int]
    position_label: str  # position as written, in `numbering`
    numbering: str  # which convention `position_label` uses
    ref_number: Optional[str]
    wild_type: str
    mutant: str
    grantham: float
    rsa: float
    priority: float
    rationale: str

    @property
    def label(self) -> str:
        return f"{self.wild_type}{self.position_label}{self.mutant}"


def _ss_at(by_ref_index: Dict[int, ResidueAnalysis], ref_index: int) -> str:
    residue = by_ref_index.get(ref_index)
    return residue.secondary_structure if residue else "-"


def _shift_boundary(
    by_ref_index: Dict[int, ResidueAnalysis],
    ref_index: int,
    step: int,
    n_residues: int,
) -> Tuple[int, bool]:
    """Walk outward until the boundary is not inside a helix or strand."""
    index = ref_index
    for _ in range(_MAX_BOUNDARY_SHIFT):
        ss = _ss_at(by_ref_index, index)
        if ss in _CUTTABLE_SS:
            return index, True
        nxt = index + step
        if nxt < 0 or nxt >= n_residues:
            return index, False
        index = nxt
    return index, False


def _member_runs(indices: Sequence[int], merge_gap: int = _MERGE_GAP) -> List[List[int]]:
    """Split patch members into runs that are close together in sequence."""
    runs: List[List[int]] = []
    for index in sorted(indices):
        if runs and index - runs[-1][-1] <= merge_gap:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def suggest_chimeras(
    patches: Sequence[Patch],
    residues: Sequence[ResidueAnalysis],
    residue_map: ResidueMap,
    structure: StructureModel,
    top_n: int = 5,
) -> List[ChimeraSuggestion]:
    """Propose domain-swap boundaries isolating each top patch.

    A conformational epitope is usually discontinuous, so each patch is proposed
    as one or more short segments rather than a single range spanning the whole
    patch - swapping everything between the first and last member would move
    most of the domain and localise nothing. Boundaries are pushed outward until
    they fall in coil/turn so a swap does not cut through a helix or strand;
    without DSSP no secondary structure is known and the suggestion says so
    instead of pretending.
    """
    by_ref_index = {r.ref_index: r for r in residues}
    n = len(residues)
    suggestions: List[ChimeraSuggestion] = []
    for patch in patches[:top_n]:
        segments: List[ChimeraSegment] = []
        for run in _member_runs([m.ref_index for m in patch.members]):
            raw_start = max(0, min(run) - _FLANK)
            raw_end = min(n - 1, max(run) + _FLANK)
            if structure.dssp_used:
                start, ok_start = _shift_boundary(by_ref_index, raw_start, -1, n)
                end, ok_end = _shift_boundary(by_ref_index, raw_end, +1, n)
                ss_ok = ok_start and ok_end
            else:
                start, end, ss_ok = raw_start, raw_end, False
            segments.append(
                ChimeraSegment(
                    start_ref_index=start,
                    end_ref_index=end,
                    start_number=residue_map.number_of(start),
                    end_number=residue_map.number_of(end),
                    ss_respected=ss_ok,
                )
            )

        if structure.dssp_used:
            note = (
                "boundaries moved to coil/turn using DSSP assignments"
                if all(s.ss_respected for s in segments)
                else "at least one boundary had no coil/turn within "
                f"{_MAX_BOUNDARY_SHIFT} residues; that swap may cut secondary structure"
            )
        else:
            note = (
                "DSSP unavailable: boundaries are the patch extent plus "
                f"{_FLANK} flanking residues and may cut through a helix or strand "
                "- check them against a structure viewer"
            )

        def inside(ref_index: int) -> bool:
            return any(
                segment.start_ref_index <= ref_index <= segment.end_ref_index
                for segment in segments
            )

        included = [
            other.patch_id
            for other in patches
            if other.patch_id != patch.patch_id
            and any(inside(m.ref_index) for m in other.members)
        ]
        n_disc = sum(
            1
            for r in residues
            if inside(r.ref_index) and r.discrimination > 0 and not r.buried
        )
        suggestions.append(
            ChimeraSuggestion(
                patch_id=patch.patch_id,
                segments=segments,
                other_patches_included=included,
                n_discriminating_included=n_disc,
                note=note,
            )
        )
    return suggestions


def suggest_mutants(
    patches: Sequence[Patch],
    dataset: Dataset,
    residue_map: ResidueMap,
    top_n: int = 5,
    max_per_patch: int = 12,
) -> List[MutantSuggestion]:
    """List reference->non-binder substitutions and their reciprocals.

    Both directions are generated: loss-of-binding mutants in the binder
    background, and the gain-of-binding mutants that are the convincing
    experiment, in each non-binder background.
    """
    reference = residue_map.reference
    non_binders = [r.name for r in dataset.non_binders]
    suggestions: List[MutantSuggestion] = []

    for patch in patches[:top_n]:
        per_patch: List[MutantSuggestion] = []
        for member in patch.members:
            wt = member.aa
            ref_position = residue_map.species_index(reference, member.ref_index)
            variants: Dict[str, List[str]] = {}
            for species in non_binders:
                residue = member.species_residues.get(species, MISSING)
                if residue in (GAP, MISSING) or residue == wt:
                    continue
                variants.setdefault(residue, []).append(species)

            for mutant, species_list in variants.items():
                d = grantham_distance(wt, mutant)
                d = 0.0 if d != d else d
                rsa = member.rsa if member.rsa == member.rsa else 0.0
                priority = d * max(rsa, 0.0) * max(member.discrimination, 0.0)
                per_patch.append(
                    MutantSuggestion(
                        patch_id=patch.patch_id,
                        direction="loss_of_binding",
                        background=reference,
                        position=ref_position,
                        # the reference is the species the structure is numbered
                        # in, so quote its own author numbering
                        position_label=member.ref_number or str(ref_position),
                        numbering=(
                            "reference structure numbering"
                            if member.ref_number
                            else f"{reference} sequence numbering"
                        ),
                        ref_number=member.ref_number,
                        wild_type=wt,
                        mutant=mutant,
                        grantham=d,
                        rsa=rsa,
                        priority=priority,
                        rationale=(
                            f"introduces the {', '.join(species_list)} residue into the "
                            f"{reference} background; loss of binding alone can also be "
                            "generic misfolding, so pair it with the reciprocal"
                        ),
                    )
                )
                for species in species_list:
                    position = residue_map.species_index(species, member.ref_index)
                    per_patch.append(
                        MutantSuggestion(
                            patch_id=patch.patch_id,
                            direction="gain_of_binding",
                            background=species,
                            position=position,
                            position_label=str(position) if position else "?",
                            numbering=f"{species} sequence numbering",
                            ref_number=member.ref_number,
                            wild_type=mutant,
                            mutant=wt,
                            grantham=d,
                            rsa=rsa,
                            priority=priority * 1.5,
                            rationale=(
                                f"restores the {reference} residue in the {species} "
                                "background; gain of binding is the convincing result"
                            ),
                        )
                    )
        per_patch.sort(key=lambda m: (-m.priority, m.direction))
        suggestions.extend(per_patch[:max_per_patch])
    return suggestions
