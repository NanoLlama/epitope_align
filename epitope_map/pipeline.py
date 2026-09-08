"""Pipeline orchestration: inputs -> alignment -> structure -> patches -> outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import __version__
from .align import (
    CONFIDENCE_CUTOFF,
    LOW_IDENTITY_MARGIN,
    Alignment,
    ResidueMap,
    align_sequences,
    alignment_confidence,
    local_identity,
)
from .glycan import DEFAULT_GLYCAN_RADIUS, GlycanAnalysis, analyse_glycosylation
from .io_seq import Dataset, InputError, build_dataset, load_binding_calls, load_sequences
from .patches import (
    DEFAULT_MIN_PATCH_SIZE,
    DEFAULT_PATCH_RADIUS,
    DEFAULT_RADIUS_SWEEP,
    Patch,
    baseline_counts,
    find_patches,
    merged_surfaces,
    promote_singletons,
    radius_sweep,
)
from .score import (
    DEFAULT_DISCRIMINATION_CUTOFF,
    indel_weight,
    panel_advice,
    ColumnScore,
    DegeneracyReport,
    ResidueAnalysis,
    assess_degeneracy,
    composite_score,
    score_alignment,
)
from .structure import StructureModel, load_structure
from .topology import (
    Segment,
    Topology,
    domain_at,
    load_domains_tsv,
    parse_topology_spec,
    parse_uniprot_features,
    require_topology,
    topology_from_range,
)
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
    topology: Optional[str] = None
    domains: Optional[str] = None
    keep_disordered: bool = False
    radius_sweep: List[float] = field(default_factory=list)
    prefer_assembly: bool = True
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
    topology: Topology = field(default_factory=Topology)
    domain_segments: List[Segment] = field(default_factory=list)
    radius_sensitivity: List[Dict[str, object]] = field(default_factory=list)
    merged_surfaces: List[Dict[str, object]] = field(default_factory=list)
    promoted_singletons: List[Patch] = field(default_factory=list)
    panel_advice: List[Dict[str, object]] = field(default_factory=list)
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
    topology: Optional[Topology] = None,
    domains: Optional[Sequence[Segment]] = None,
) -> List[ResidueAnalysis]:
    """Fuse sequence scores with structural values into one row per residue.

    Residues an antibody cannot reach are hard-excluded here: their scores are
    zeroed, not merely down-weighted, so they cannot surface anywhere downstream
    as evidence.
    """
    topology = topology or Topology()
    domains = list(domains or [])
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
            indel_length=score.indel_length,
            has_missing=score.has_missing,
            modelled=record is not None,
            in_ectodomain=position.in_ectodomain,
            notes=list(score.notes),
        )
        sequence_position = position.ref_index + 1
        row.topology = topology.kind_at(sequence_position)
        domain = domain_at(domains, sequence_position)
        row.domain = domain.description if domain else ""
        row.accessible = position.in_ectodomain and topology.is_accessible(
            sequence_position
        )
        if record is not None:
            row.sasa = record.sasa
            row.rsa = record.rsa_raw if config.keep_disordered else record.rsa
            row.plddt = record.plddt
            row.secondary_structure = record.secondary_structure
            row.centroid = record.centroid
            row.in_disordered_region = record.in_disordered_region
        row.buried = not (row.rsa == row.rsa and row.rsa >= config.rsa_cutoff)

        if row.involves_gap:
            # believe a long loop insertion; discount a one-residue gap inside a
            # helix, which is usually the aligner's guess rather than biology
            weight = indel_weight(row.indel_length, row.secondary_structure)
            row.discrimination *= weight
            if weight < 1.0:
                row.notes.append(
                    f"indel of {row.indel_length} residue(s) down-weighted "
                    f"x{weight:.2f} (short, or inside a secondary-structure element)"
                )

        if not row.accessible:
            # an antibody cannot reach it, so it is not a candidate at any score
            row.discrimination = 0.0
            row.max_grantham = 0.0
            row.mask_reasons.append(
                "outside_ectodomain"
                if not row.in_ectodomain and topology.is_accessible(sequence_position)
                else "outside_topology"
            )
            if row.topology not in ("unknown", "extracellular"):
                row.notes.append(
                    f"{row.topology.replace('_', ' ')} - not reachable by an antibody"
                )

        composite, exposure, confidence = composite_score(
            row.discrimination, row.rsa, row.plddt
        )
        row.exposure_weight = exposure
        row.confidence_weight = confidence
        row.composite = composite

        if not row.modelled:
            row.mask_reasons.append("not_modelled_in_structure")
        elif row.buried:
            row.mask_reasons.append("buried")
        if record is not None and record.buried_by_context:
            row.mask_reasons.append("assembly_interface")
        if row.in_disordered_region and not config.keep_disordered:
            row.mask_reasons.append("disordered_region")
        rows.append(row)
    return rows


def _apply_alignment_reliability(
    rows: Sequence[ResidueAnalysis],
    confidence: Dict[int, float],
    identity_windows: Dict[int, float],
) -> None:
    """Attach per-residue alignment reliability and flag the shaky windows."""
    values = [v for v in identity_windows.values() if v == v]
    average = sum(values) / len(values) if values else 0.0
    for row in rows:
        row.alignment_confidence = confidence.get(row.ref_index, float("nan"))
        row.local_identity = identity_windows.get(row.ref_index, float("nan"))
        if row.local_identity == row.local_identity and (
            row.local_identity < average - LOW_IDENTITY_MARGIN
        ):
            row.low_identity_window = True
            row.notes.append(
                f"local identity {row.local_identity:.0f}% against a chain average "
                f"of {average:.0f}%: the aligner has little to go on here"
            )
        if row.alignment_confidence == row.alignment_confidence and (
            row.alignment_confidence < CONFIDENCE_CUTOFF
        ):
            row.notes.append(
                f"alignment confidence {row.alignment_confidence:.2f}: different "
                "alignment methods disagree about which residue of the other "
                "species corresponds to this one"
            )


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


def resolve_topology(
    config: RunConfig, dataset: Dataset, reference: str
) -> Tuple[Topology, List[Segment]]:
    """Work out which part of the chain faces the outside, or ask the user.

    Explicit ``--topology`` wins; otherwise the reference's UniProt features are
    used when it was fetched by accession; otherwise an ``--ectodomain`` range
    stands in for the extracellular segment. When none of those is available the
    caller must stop, because defaulting to the whole chain ranks the inside of
    the cell.
    """
    domains: List[Segment] = []
    reference_record = dataset.get(reference)

    if reference_record.features:
        derived, domains = parse_uniprot_features(reference_record.features)
    else:
        derived = Topology()

    if config.topology:
        topology = parse_topology_spec(config.topology, length=len(reference_record.sequence))
    elif derived.known:
        topology = derived
    elif config.ectodomain and config.ectodomain_numbering == "sequence":
        topology = topology_from_range(*config.ectodomain)
    else:
        topology = derived  # unknown; --ectodomain in structure numbering may still cover us

    if config.domains:
        domains = load_domains_tsv(Path(config.domains))
    return topology, domains


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

    topology, domain_segments = resolve_topology(config, dataset, reference)
    if not (topology.known or topology.whole_chain or config.ectodomain):
        require_topology(
            None, len(dataset.get(reference).sequence), reference
        )

    structure = load_structure(
        config.structure,
        chain_id=config.chain,
        context_spec=config.assembly_context,
        context_chains=config.context_chains,
        cache_dir=config.cache_dir,
        run_dssp=config.run_dssp,
        prefer_assembly=config.prefer_assembly,
    )
    residue_map = ResidueMap.build(
        alignment,
        structure,
        ectodomain=config.ectodomain,
        ectodomain_numbering=config.ectodomain_numbering,
        mismatch_tolerance=config.mismatch_tolerance,
    )

    confidence, confidence_notes = alignment_confidence(
        alignment, list(dataset.records), threads=config.threads
    )
    identity_windows = local_identity(alignment)

    column_scores = score_alignment(residue_map, dataset)
    rows = build_residue_table(
        residue_map, column_scores, structure, config, topology, domain_segments
    )
    _apply_alignment_reliability(rows, confidence, identity_windows)
    accessible = {row.ref_index for row in rows if row.accessible}

    glycans = analyse_glycosylation(
        residue_map,
        dataset,
        structure,
        radius=config.glycan_radius,
        is_accessible=lambda ref_index: ref_index in accessible,
        topology_kind=lambda ref_index: topology.kind_at(ref_index + 1),
        near_boundary=lambda ref_index: topology.near_boundary(ref_index + 1),
    )
    _apply_glycan_flags(rows, glycans)
    occlusion_warnings = _apply_occluding_chains(rows, residue_map, config)

    # calibrate on the residues an antibody could actually reach: including the
    # cytoplasm in the background rate flatters the observed signal
    degeneracy = assess_degeneracy(
        [s for s in column_scores if s.ref_index in accessible] or column_scores,
        dataset,
        cutoff=config.discrimination_cutoff,
    )
    patches, singletons = find_patches(
        rows,
        discrimination_cutoff=config.discrimination_cutoff,
        radius=config.patch_radius,
        min_size=config.min_patch_size,
        method=config.cluster_method,
        rsa_cutoff=config.rsa_cutoff,
    )
    promoted, singletons = promote_singletons(singletons, patches)
    surfaces = merged_surfaces(patches, promoted)
    sweep: List[Dict[str, object]] = []
    if config.radius_sweep:
        radii = (
            list(DEFAULT_RADIUS_SWEEP)
            if config.radius_sweep == ["auto"]
            else config.radius_sweep
        )
        sweep = radius_sweep(
            rows,
            discrimination_cutoff=config.discrimination_cutoff,
            radii=radii,
            min_size=config.min_patch_size,
            method=config.cluster_method,
            rsa_cutoff=config.rsa_cutoff,
        )
    advice = panel_advice(
        residue_map, dataset, cutoff=config.discrimination_cutoff, accessible=accessible
    )
    chimeras = suggest_chimeras(patches, rows, residue_map, structure, top_n=config.top_n)
    mutants = suggest_mutants(patches, dataset, residue_map, top_n=config.top_n)
    counts = baseline_counts(rows, config.discrimination_cutoff)

    if topology.known and not topology.whole_chain:
        excluded = sum(1 for row in rows if not row.accessible)
        warnings_topology = [
            f"topology ({topology.source}): {topology.summary()}. {excluded} "
            "residue(s) are not reachable by an antibody and were excluded from "
            "scoring entirely"
        ] + list(topology.warnings)
    elif topology.whole_chain:
        warnings_topology = [
            "the whole chain was declared accessible (--topology whole-chain); "
            "no membrane topology was applied"
        ]
    else:
        warnings_topology = [
            "no membrane topology was available, so only the supplied "
            "--ectodomain range restricts the analysis; if this is a membrane "
            "protein, check that the range excludes the transmembrane helix and "
            "the cytoplasmic tail"
        ]

    unreliable = [
        r for r in rows
        if r.accessible
        and r.alignment_confidence == r.alignment_confidence
        and r.alignment_confidence < CONFIDENCE_CUTOFF
    ]
    if unreliable:
        confidence_notes.append(
            f"{len(unreliable)} accessible position(s) have alignment confidence "
            f"below {CONFIDENCE_CUTOFF}. Point mutants there name a residue "
            "equivalence the aligner guessed, so they are marked unverified; "
            "chimera-level suggestions for the same region still stand"
        )

    warnings_ = (
        warnings_topology
        + confidence_notes
        + list(dataset.warnings)
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
        topology=topology,
        domain_segments=domain_segments,
        radius_sensitivity=sweep,
        merged_surfaces=surfaces,
        promoted_singletons=promoted,
        panel_advice=advice,
        warnings=warnings_,
    )
