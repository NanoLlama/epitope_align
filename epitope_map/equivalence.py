"""Residue equivalences derived from superposed structures rather than sequence.

A point mutant names one residue of a non-binder as the counterpart of one
residue of the reference. In a low-identity, indel-bearing loop that pairing is
whatever the aligner guessed, and the region can be right while the construct is
wrong. Where a structure of the non-binder exists, superposing the two and taking
the spatially nearest residue answers the same question with geometry, which is
more reliable in exactly those loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .align import GAP, ResidueMap
from .structure import StructureModel, distance, load_structure

#: A reference CA and a non-binder CA further apart than this after superposition
#: are not the same position, whatever the alignment says.
MAX_EQUIVALENCE_DISTANCE = 4.0


@dataclass
class StructuralEquivalence:
    """Reference residue index -> that species' 1-based sequence position."""

    species: str
    mapping: Dict[int, int] = field(default_factory=dict)
    rmsd: float = float("nan")
    n_superposed: int = 0
    disagreements: Dict[int, Tuple[Optional[int], int]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.mapping) and self.n_superposed >= 20


def _sequence_index_by_key(model: StructureModel) -> Dict[object, int]:
    return {record.key: index for index, record in enumerate(model.residues)}


def superpose(
    reference: StructureModel,
    other: StructureModel,
    pairs: Sequence[Tuple[int, int]],
) -> Tuple[List[Tuple[float, float, float]], float]:
    """Superpose ``other`` onto ``reference`` using anchor residue pairs.

    Returns the transformed coordinates of every residue of ``other`` and the
    RMSD over the anchors.
    """
    from Bio.PDB import Superimposer
    from Bio.PDB.Atom import Atom

    fixed, moving = [], []
    for ref_index, other_index in pairs:
        a = reference.residues[ref_index].ca
        b = other.residues[other_index].ca
        if a is None or b is None:
            continue
        fixed.append(Atom("CA", a, 0.0, 1.0, " ", "CA", 0, "C"))
        moving.append(Atom("CA", b, 0.0, 1.0, " ", "CA", 0, "C"))
    if len(fixed) < 3:
        raise ValueError("need at least three anchor pairs to superpose")

    superimposer = Superimposer()
    superimposer.set_atoms(fixed, moving)
    rotation, translation = superimposer.rotran

    transformed: List[Tuple[float, float, float]] = []
    for record in other.residues:
        point = record.ca or record.centroid
        if point is None:
            transformed.append(None)
            continue
        x, y, z = point
        moved = [
            x * rotation[0][0] + y * rotation[1][0] + z * rotation[2][0] + translation[0],
            x * rotation[0][1] + y * rotation[1][1] + z * rotation[2][1] + translation[1],
            x * rotation[0][2] + y * rotation[1][2] + z * rotation[2][2] + translation[2],
        ]
        transformed.append(tuple(moved))
    return transformed, float(superimposer.rms)


def structural_equivalence(
    residue_map: ResidueMap,
    reference_structure: StructureModel,
    species: str,
    species_structure: StructureModel,
    max_distance: float = MAX_EQUIVALENCE_DISTANCE,
) -> StructuralEquivalence:
    """Pair reference residues with a non-binder's by 3D proximity.

    The sequence alignment provides the anchors for the superposition - it is
    reliable in the conserved frame, which is all a superposition needs - and
    geometry then decides the pairings inside the loops, where it is not.
    """
    result = StructuralEquivalence(species=species)

    # anchors: confidently aligned, modelled positions in both structures
    ref_by_index = {}
    for position in residue_map.positions():
        if position.key is None:
            continue
        index = _sequence_index_by_key(reference_structure).get(position.key)
        if index is not None:
            ref_by_index[position.ref_index] = index

    species_sequence = residue_map.alignment.sequences[species].replace(GAP, "")
    species_offset = _offset_of(species_structure.sequence, species_sequence)
    if species_offset is None:
        result.warnings.append(
            f"the structure supplied for {species} does not match its sequence, "
            "so structural equivalences were not used"
        )
        return result

    anchors: List[Tuple[int, int]] = []
    for ref_index, ref_structure_index in ref_by_index.items():
        position = residue_map.species_index(species, ref_index)
        if position is None:
            continue
        if (
            residue_map.aa_at(ref_index)
            != residue_map.residue_at(species, ref_index)
        ):
            continue  # anchor on identities only: they are the reliable part
        structure_index = position - 1 - species_offset
        if 0 <= structure_index < len(species_structure.residues):
            anchors.append((ref_structure_index, structure_index))

    if len(anchors) < 20:
        result.warnings.append(
            f"only {len(anchors)} confident anchor(s) between the reference and "
            f"the {species} structure - too few to superpose reliably, so "
            "sequence equivalences were kept"
        )
        return result

    try:
        transformed, rmsd = superpose(reference_structure, species_structure, anchors)
    except Exception as exc:  # pragma: no cover - defensive
        result.warnings.append(f"superposition of {species} failed: {exc}")
        return result

    result.rmsd = rmsd
    result.n_superposed = len(anchors)

    for ref_index, ref_structure_index in ref_by_index.items():
        origin = reference_structure.residues[ref_structure_index].ca
        if origin is None:
            continue
        best_index, best_distance = None, max_distance
        for index, point in enumerate(transformed):
            if point is None:
                continue
            gap = distance(origin, point)
            if gap < best_distance:
                best_index, best_distance = index, gap
        if best_index is None:
            continue
        sequence_position = best_index + species_offset + 1
        result.mapping[ref_index] = sequence_position
        from_sequence = residue_map.species_index(species, ref_index)
        if from_sequence != sequence_position:
            result.disagreements[ref_index] = (from_sequence, sequence_position)

    if result.disagreements:
        result.warnings.append(
            f"structural and sequence equivalences disagree at "
            f"{len(result.disagreements)} position(s) for {species} "
            f"(superposition RMSD {rmsd:.2f} A over {len(anchors)} residues). "
            "The structural pairing is used for mutant design there; those "
            "positions are exactly where a sequence-derived mutant would have "
            "named the wrong residue"
        )
    return result


def _offset_of(structure_sequence: str, full_sequence: str) -> Optional[int]:
    """Where the modelled stretch starts within the full sequence."""
    if not structure_sequence:
        return None
    index = full_sequence.find(structure_sequence)
    if index >= 0:
        return index
    # tolerate a few mismatches (modified residues, engineered mutations)
    best_index, best_score = None, 0
    window = len(structure_sequence)
    for start in range(0, max(1, len(full_sequence) - window + 1)):
        chunk = full_sequence[start : start + window]
        score = sum(1 for a, b in zip(chunk, structure_sequence) if a == b)
        if score > best_score:
            best_index, best_score = start, score
    if best_index is not None and best_score >= 0.8 * len(structure_sequence):
        return best_index
    return None


def load_species_structures(
    spec: Sequence[str], cache_dir: Optional[Path] = None
) -> Dict[str, StructureModel]:
    """Parse ``species=path`` entries into loaded structures."""
    out: Dict[str, StructureModel] = {}
    for item in spec:
        if "=" not in item:
            raise ValueError(
                f"expected 'species=structure', got {item!r}"
            )
        species, source = item.split("=", 1)
        out[species.strip()] = load_structure(
            source.strip(), cache_dir=cache_dir, run_dssp=False
        )
    return out
