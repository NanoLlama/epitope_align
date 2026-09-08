"""N-linked glycosylation sequon scanning and proximity occlusion flags."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .align import GAP, ResidueMap
from .io_seq import Dataset
from .structure import StructureModel, distance

#: A glycan reaches well beyond its attachment point; residues within this
#: radius of a differential sequon asparagine are flagged as possibly occluded.
DEFAULT_GLYCAN_RADIUS = 12.0


@dataclass
class Sequon:
    """One N-X-S/T sequon in one species, expressed in reference coordinates."""

    species: str
    column: int
    ref_index: Optional[int]
    ref_number: Optional[str]
    motif: str
    present_in: List[str] = field(default_factory=list)
    absent_in: List[str] = field(default_factory=list)

    @property
    def differential(self) -> bool:
        return bool(self.present_in) and bool(self.absent_in)


@dataclass
class GlycanAnalysis:
    sequons: List[Sequon] = field(default_factory=list)
    differential: List[Sequon] = field(default_factory=list)
    occluded: Dict[int, List[str]] = field(default_factory=dict)
    radius: float = DEFAULT_GLYCAN_RADIUS
    unplaced: List[Sequon] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _ungapped_positions(aligned: str) -> List[int]:
    """Alignment column of each ungapped residue in a species' sequence."""
    return [i for i, c in enumerate(aligned) if c != GAP]


def scan_sequons(aligned: str) -> List[int]:
    """Return the alignment column of the N in every N-X-S/T (X != P) sequon.

    The scan runs on the ungapped sequence - gaps in the alignment must not
    break or create a sequon - and the result is mapped back to columns.
    """
    columns = _ungapped_positions(aligned)
    seq = "".join(aligned[i] for i in columns)
    hits: List[int] = []
    for i in range(len(seq) - 2):
        if seq[i] != "N":
            continue
        x, st = seq[i + 1], seq[i + 2]
        if x == "P":
            continue
        if st in ("S", "T"):
            hits.append(columns[i])
    return hits


def analyse_glycosylation(
    residue_map: ResidueMap,
    dataset: Dataset,
    structure: StructureModel,
    radius: float = DEFAULT_GLYCAN_RADIUS,
) -> GlycanAnalysis:
    """Find species-specific sequons and flag residues a glycan could occlude."""
    analysis = GlycanAnalysis(radius=radius)
    alignment = residue_map.alignment

    per_species: Dict[str, Set[int]] = {
        name: set(scan_sequons(seq)) for name, seq in alignment.sequences.items()
    }
    scored = [r.name for r in dataset.scored]
    binders = {r.name for r in dataset.binders}
    non_binders = {r.name for r in dataset.non_binders}

    all_columns = sorted({c for name in scored for c in per_species[name]})
    coords = _reference_coordinates(residue_map, structure)

    for column in all_columns:
        present = [n for n in scored if column in per_species[n]]
        absent = [n for n in scored if column not in per_species[n]]
        ref_index = residue_map.ref_index_of_column(column)
        motif = "".join(
            alignment.sequences[present[0]][c]
            for c in _motif_columns(alignment.sequences[present[0]], column)
        )
        sequon = Sequon(
            species=",".join(present),
            column=column,
            ref_index=ref_index,
            ref_number=residue_map.number_of(ref_index) if ref_index is not None else None,
            motif=motif,
            present_in=present,
            absent_in=absent,
        )
        analysis.sequons.append(sequon)

        binder_present = set(present) & binders
        non_binder_present = set(present) & non_binders
        clean_split = (
            (binder_present == binders and not non_binder_present)
            or (non_binder_present == non_binders and not binder_present)
        )
        if sequon.differential and clean_split:
            analysis.differential.append(sequon)

    for sequon in analysis.differential:
        if sequon.ref_index is None or sequon.ref_index not in coords:
            analysis.unplaced.append(sequon)
            continue
        origin = coords[sequon.ref_index]
        for ref_index, xyz in coords.items():
            if distance(origin, xyz) <= radius:
                label = (
                    f"within {radius:.0f} A of a sequon at reference "
                    f"{sequon.ref_number} present in {sorted(set(sequon.present_in))} "
                    f"and absent in {sorted(set(sequon.absent_in))}"
                )
                analysis.occluded.setdefault(ref_index, []).append(label)

    if analysis.differential:
        analysis.warnings.append(
            f"{len(analysis.differential)} differential N-glycosylation sequon(s) "
            "separate binders from non-binders. A glycan acts at a distance: the "
            f"causative difference may sit outside the antibody footprint, so all "
            f"surface residues within {radius:.0f} A are flagged as possibly occluded."
        )
    if analysis.unplaced:
        analysis.warnings.append(
            f"{len(analysis.unplaced)} differential sequon(s) fall on alignment "
            "columns with no modelled reference residue, so no proximity flags "
            "could be computed for them"
        )
    if structure.is_alphafold:
        analysis.warnings.append(
            "AlphaFold models carry no glycans, so the modelled surface is more "
            "accessible than the real one"
        )
    return analysis


def _motif_columns(aligned: str, start_column: int) -> List[int]:
    columns = [i for i, c in enumerate(aligned) if c != GAP]
    try:
        index = columns.index(start_column)
    except ValueError:  # pragma: no cover - defensive
        return [start_column]
    return columns[index : index + 3]


def _reference_coordinates(
    residue_map: ResidueMap, structure: StructureModel
) -> Dict[int, Sequence[float]]:
    by_key = structure.by_key()
    coords: Dict[int, Sequence[float]] = {}
    for position in residue_map.positions():
        if position.key is None:
            continue
        record = by_key.get(position.key)
        if record is None or record.centroid is None:
            continue
        coords[position.ref_index] = record.centroid
    return coords
