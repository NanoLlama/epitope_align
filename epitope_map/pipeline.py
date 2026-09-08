"""Pipeline orchestration: inputs -> alignment -> structure -> patches -> outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import __version__
from .align import Alignment, ResidueMap, align_sequences
from .glycan import DEFAULT_GLYCAN_RADIUS, GlycanAnalysis, analyse_glycosylation
from .io_seq import Dataset, InputError, build_dataset, load_binding_calls, load_sequences
from .patches import (
    DEFAULT_MIN_PATCH_SIZE,
    DEFAULT_PATCH_RADIUS,
    Patch,
    baseline_counts,
    find_patches,
)
from .score import (
    DEFAULT_DISCRIMINATION_CUTOFF,
    ColumnScore,
    DegeneracyReport,
    ResidueAnalysis,
    assess_degeneracy,
    composite_score,
    score_alignment,
)
from .structure import StructureModel, load_structure
from .suggest import ChimeraSuggestion, MutantSuggestion, suggest_chimeras, suggest_mutants

DEFAULT_RSA_CUTOFF = 0.20


@dataclass
class RunConfig:
    """Everything the pipeline needs; the CLI and config file both build one."""

    sequences: str
    binding: str
    reference: str
    structure: str
    outdir: Path = Path("results")
    ectodomain: Optional[Tuple[int, int]] = None
    ectodomain_numbering: str = "structure"
    chain: Optional[str] = None
    assembly_context: Optional[str] = None
    context_chains: List[str] = field(default_factory=list)
    occluding_chains: List[str] = field(default_factory=list)
    occluding_structure: Optional[str] = None
    rsa_cutoff: float = DEFAULT_RSA_CUTOFF
    discrimination_cutoff: float = DEFAULT_DISCRIMINATION_CUTOFF
    patch_radius: float = DEFAULT_PATCH_RADIUS
    min_patch_size: int = DEFAULT_MIN_PATCH_SIZE
    glycan_radius: float = DEFAULT_GLYCAN_RADIUS
    cluster_method: str = "graph"
    aligner: str = "auto"
    threads: int = 1
    top_n: int = 5
    mismatch_tolerance: float = 0.05
    run_dssp: bool = True
    cache_dir: Optional[Path] = None

    def as_dict(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, value in self.__dict__.items():
            if value is None or value == [] :
                out[key] = "-"
            elif isinstance(value, tuple):
                out[key] = f"{value[0]}-{value[1]}"
            elif isinstance(value, list):
                out[key] = ",".join(str(v) for v in value)
            else:
                out[key] = str(value)
        return out


@dataclass
class RunResult:
    """Everything the report writer needs."""

    config: RunConfig
    dataset: Dataset
    alignment: Alignment
    residue_map: ResidueMap
    structure: StructureModel
    residues: List[ResidueAnalysis]
    column_scores: List[ColumnScore]
    patches: List[Patch]
    singletons: List[Patch]
    degeneracy: DegeneracyReport
    glycans: GlycanAnalysis
    chimeras: List[ChimeraSuggestion]
    mutants: List[MutantSuggestion]
    counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)
    version: str = __version__


def _load_inputs(config: RunConfig) -> Tuple[Dataset, str]:
    records = load_sequences(config.sequences, cache_dir=config.cache_dir)
    calls = load_binding_calls(Path(config.binding))
    return build_dataset(records, calls, config.reference)


def build_residue_table(
    residue_map: ResidueMap,
    column_scores: Sequence[ColumnScore],
    structure: StructureModel,
    config: RunConfig,
) -> List[ResidueAnalysis]:
    """Fuse sequence scores with structural values into one row per residue."""
    by_key = structure.by_key()
    rows: List[ResidueAnalysis] = []
    for position, score in zip(residue_map.positions(), column_scores):
        record = by_key.get(position.key) if position.key else None
        row = ResidueAnalysis(
            ref_index=position.ref_index,
            column=position.column,
            aa=position.aa,
            ref_number=position.number,
            chain=structure.chain_id if record else None,
            species_residues=dict(score.residues),
            discrimination=score.discrimination,
            pattern_consistency=score.pattern_consistency,
            max_grantham=score.max_grantham,
            conserved_in_binders=score.conserved_in_binders,
            involves_gap=score.involves_gap,
            has_missing=score.has_missing,
            modelled=record is not None,
            in_ectodomain=position.in_ectodomain,
            notes=list(score.notes),
        )
        if record is not None:
            row.sasa = record.sasa
            row.rsa = record.rsa
            row.plddt = record.plddt
            row.secondary_structure = record.secondary_structure
            row.centroid = record.centroid
        row.buried = not (row.rsa == row.rsa and row.rsa >= config.rsa_cutoff)

        composite, exposure, confidence = composite_score(
            row.discrimination, row.rsa, row.plddt
        )
        row.exposure_weight = exposure
        row.confidence_weight = confidence
        row.composite = composite

        if not row.in_ectodomain:
            row.mask_reasons.append("outside_ectodomain")
        if not row.modelled:
            row.mask_reasons.append("not_modelled_in_structure")
        elif row.buried:
            row.mask_reasons.append("buried")
        if record is not None and record.buried_by_context:
            row.mask_reasons.append("assembly_interface")
        rows.append(row)
    return rows


def _apply_glycan_flags(rows: Sequence[ResidueAnalysis], glycans: GlycanAnalysis) -> None:
    sequon_indices = {s.ref_index for s in glycans.differential if s.ref_index is not None}
    for row in rows:
        flags = glycans.occluded.get(row.ref_index)
        if flags:
            row.glycan_flags.extend(flags)
        if row.ref_index in sequon_indices:
            row.is_sequon_asn = True
            row.notes.append(
                "asparagine of a differential N-glycosylation sequon: the glycan "
                "itself, not this side chain, may be what the antibody senses"
            )


def _apply_occluding_chains(
    rows: Sequence[ResidueAnalysis],
    residue_map: ResidueMap,
    config: RunConfig,
) -> List[str]:
    """Mask residues occluded by a known interacting partner."""
    if not config.occluding_chains and not config.occluding_structure:
        return []
    # the reference structure again, this time with the partner alongside it, so
    # a residue that loses accessibility is one the partner covers
    occluding = load_structure(
        config.structure,
        chain_id=config.chain,
        context_spec=config.occluding_structure,
        context_chains=config.occluding_chains,
        cache_dir=config.cache_dir,
        run_dssp=False,
    )
    by_key = occluding.by_key()
    masked = 0
    for row in rows:
        key = residue_map.key_of(row.ref_index)
        record = by_key.get(key) if key else None
        if record is not None and record.buried_by_context:
            if "occluded_by_partner" not in row.mask_reasons:
                row.mask_reasons.append("occluded_by_partner")
                masked += 1
    if masked:
        return [
            f"{masked} residue(s) masked as occluded by the supplied interacting "
            f"partner chains ({', '.join(config.occluding_chains) or 'all extra chains'})"
        ]
    return []


def run_pipeline(config: RunConfig) -> RunResult:
    """Run every stage and return the assembled result (no files written)."""
    dataset, reference = _load_inputs(config)
    config.reference = reference

    alignment = align_sequences(
        list(dataset.records),
        reference=reference,
        method=config.aligner,
        threads=config.threads,
    )
    # the degeneracy check needs the aligned sequences to build its tree
    setattr(dataset, "_alignment_sequences", alignment.sequences)

    structure = load_structure(
        config.structure,
        chain_id=config.chain,
        context_spec=config.assembly_context,
        context_chains=config.context_chains,
        cache_dir=config.cache_dir,
        run_dssp=config.run_dssp,
    )
    residue_map = ResidueMap.build(
        alignment,
        structure,
        ectodomain=config.ectodomain,
        ectodomain_numbering=config.ectodomain_numbering,
        mismatch_tolerance=config.mismatch_tolerance,
    )

    column_scores = score_alignment(residue_map, dataset)
    rows = build_residue_table(residue_map, column_scores, structure, config)

    glycans = analyse_glycosylation(
        residue_map, dataset, structure, radius=config.glycan_radius
    )
    _apply_glycan_flags(rows, glycans)
    occlusion_warnings = _apply_occluding_chains(rows, residue_map, config)

    degeneracy = assess_degeneracy(
        column_scores, dataset, cutoff=config.discrimination_cutoff
    )
    patches, singletons = find_patches(
        rows,
        discrimination_cutoff=config.discrimination_cutoff,
        radius=config.patch_radius,
        min_size=config.min_patch_size,
        method=config.cluster_method,
    )
    chimeras = suggest_chimeras(patches, rows, residue_map, structure, top_n=config.top_n)
    mutants = suggest_mutants(patches, dataset, residue_map, top_n=config.top_n)
    counts = baseline_counts(rows, config.discrimination_cutoff)

    warnings_ = (
        list(dataset.warnings)
        + list(alignment.warnings)
        + list(residue_map.warnings)
        + list(structure.warnings)
        + list(glycans.warnings)
        + occlusion_warnings
    )
    if degeneracy.clade_split:
        warnings_.append(
            "TWO-CLADE DEGENERACY: " + degeneracy.clade_split_detail
        )
    if not dataset.binders or not dataset.non_binders:  # pragma: no cover - guarded earlier
        raise InputError("both a binder and a non-binder are required")

    return RunResult(
        config=config,
        dataset=dataset,
        alignment=alignment,
        residue_map=residue_map,
        structure=structure,
        residues=rows,
        column_scores=column_scores,
        patches=patches,
        singletons=singletons,
        degeneracy=degeneracy,
        glycans=glycans,
        chimeras=chimeras,
        mutants=mutants,
        counts=counts,
        warnings=warnings_,
    )
