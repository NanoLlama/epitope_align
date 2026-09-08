"""Output writers: TSVs, report.md, alignment.fasta and the PyMOL session."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .patches import TYPICAL_EPITOPE_BSA, TYPICAL_EPITOPE_RESIDUES, Patch
from .pipeline import RunResult
from .score import MISSING

_PATCH_COLORS = [
    "red", "orange", "yellow", "green", "cyan", "blue", "magenta", "salmon",
    "lime", "slate",
]

CAVEATS = [
    "The causative difference need not lie inside the antibody footprint. A "
    "substitution just outside it can shift a loop, and a glycan occludes at a "
    "distance - which is why sequon-proximal residues are flagged here.",
    "Some discriminating residues inside the true epitope are energetically "
    "neutral. Membership in a patch is not evidence of a contribution to binding.",
    "Two-clade comparisons have limited resolving power. More species with mixed "
    "binding outcomes is the single biggest improvement available - each "
    "additional informative species roughly halves the candidate set.",
    "AlphaFold surfaces are unglycosylated, and loop conformations in low-pLDDT "
    "regions are unreliable.",
    "This output is a ranked hypothesis list for experiment design, not a "
    "prediction of the epitope.",
]


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return f"{value:.{digits}f}"


def write_all(result: RunResult, outdir: Path) -> Dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "residues": write_residues(result, outdir / "residues.tsv"),
        "chimeras": write_chimeras(result, outdir / "chimeras.tsv"),
        "patches": write_patches(result, outdir / "patches.tsv"),
        "mutants": write_mutants(result, outdir / "mutants.tsv"),
        "alignment": write_alignment(result, outdir / "alignment.fasta"),
        "pymol": write_pymol(result, outdir / "session.pml"),
        "report": write_report(result, outdir / "report.md"),
    }
    return paths


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


def residues_dataframe(result: RunResult) -> pd.DataFrame:
    species = list(result.alignment.sequences.keys())
    rows: List[Dict[str, object]] = []
    for residue in result.residues:
        row: Dict[str, object] = {
            "ref_number": residue.ref_number or "",
            "ref_index_1based": residue.ref_index + 1,
            "chain": residue.chain or "",
            "aa": residue.aa,
            "alignment_column": residue.column + 1,
        }
        for name in species:
            row[f"aa_{name}"] = residue.species_residues.get(name, MISSING)
        row.update(
            {
                "discrimination": round(residue.discrimination, 4),
                "pattern_consistency": round(residue.pattern_consistency, 4),
                "max_grantham": round(residue.max_grantham, 1),
                "conserved_in_binders": residue.conserved_in_binders,
                "indel": residue.involves_gap,
                "missing_species": residue.has_missing,
                "sasa": _fmt(residue.sasa, 1),
                "rsa": _fmt(residue.rsa, 3),
                "buried": residue.buried,
                "plddt": _fmt(residue.plddt, 1),
                "plddt_flag": residue.plddt_flag,
                "secondary_structure": residue.secondary_structure,
                "glycan_flag": "; ".join(residue.glycan_flags),
                "sequon_asn": residue.is_sequon_asn,
                "exposure_weight": round(residue.exposure_weight, 4),
                "confidence_weight": round(residue.confidence_weight, 4),
                "composite": round(residue.composite, 4),
                "mask_reasons": ";".join(residue.mask_reasons),
                "patch_id": residue.patch_id or "",
                "notes": " | ".join(residue.notes),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_residues(result: RunResult, path: Path) -> Path:
    residues_dataframe(result).to_csv(path, sep="\t", index=False)
    return path


def patches_dataframe(result: RunResult, patches: Optional[Sequence[Patch]] = None) -> pd.DataFrame:
    patches = result.patches if patches is None else patches
    rows = []
    for patch in patches:
        centroid = patch.centroid
        rows.append(
            {
                "patch_id": patch.patch_id,
                "rank_raw": patch.rank_raw,
                "rank_normalized": patch.rank_normalized,
                "n_residues": patch.size,
                "residues": ",".join(patch.residue_labels),
                "ref_numbers": ",".join(
                    m.ref_number or str(m.ref_index + 1) for m in patch.members
                ),
                "total_score": round(patch.total_score, 4),
                "mean_score": round(patch.mean_score, 4),
                "normalized_score": round(patch.normalized_score, 4),
                "n_discriminating": patch.n_discriminating,
                "max_grantham": round(patch.max_grantham, 1),
                "mean_rsa": _fmt(patch.mean_rsa, 3),
                "accessible_area_A2": _fmt(patch.surface_area, 1),
                "spread_A": _fmt(patch.spread, 1),
                "n_indels": patch.n_indels,
                "n_glycan_flagged": patch.n_glycan_flagged,
                "centroid_x": _fmt(centroid[0], 2),
                "centroid_y": _fmt(centroid[1], 2),
                "centroid_z": _fmt(centroid[2], 2),
                "flags": " | ".join(patch.flags),
                "rationale": patch.rationale(),
            }
        )
    return pd.DataFrame(rows)


def write_patches(result: RunResult, path: Path) -> Path:
    frame = patches_dataframe(result)
    singles = patches_dataframe(result, result.singletons)
    if not singles.empty:
        singles = singles.copy()
        singles["patch_id"] = singles["patch_id"].apply(lambda p: f"{p}(singleton)")
        frame = pd.concat([frame, singles], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)
    return path


def write_mutants(result: RunResult, path: Path) -> Path:
    rows = [
        {
            "patch_id": m.patch_id,
            "direction": m.direction,
            "background_species": m.background,
            "mutation": m.label,
            "numbering": m.numbering,
            "position_in_background_sequence": m.position if m.position is not None else "",
            "reference_number": m.ref_number or "",
            "wild_type": m.wild_type,
            "mutant": m.mutant,
            "grantham": round(m.grantham, 1),
            "rsa": _fmt(m.rsa, 3),
            "priority": round(m.priority, 3),
            "rationale": m.rationale,
        }
        for m in result.mutants
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def write_chimeras(result: RunResult, path: Path) -> Path:
    rows = []
    for chimera in result.chimeras:
        for index, segment in enumerate(chimera.segments, start=1):
            rows.append(
                {
                    "patch_id": chimera.patch_id,
                    "segment": f"{index}/{len(chimera.segments)}",
                    "start_reference_number": segment.start_number or "",
                    "end_reference_number": segment.end_number or "",
                    "length": segment.length,
                    "secondary_structure_respected": segment.ss_respected,
                    "patch_total_swapped_residues": chimera.length,
                    "other_patches_included": ",".join(chimera.other_patches_included),
                    "discriminating_positions_inside": chimera.n_discriminating_included,
                    "note": chimera.note,
                }
            )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def write_alignment(result: RunResult, path: Path) -> Path:
    path.write_text(result.alignment.to_fasta())
    return path


# --------------------------------------------------------------------------
# PyMOL
# --------------------------------------------------------------------------


def _pymol_selection(patch: Patch, chain: str) -> str:
    numbers = "+".join(m.ref_number for m in patch.members if m.ref_number)
    return f"chain {chain} and resi {numbers}"


def write_pymol(result: RunResult, path: Path) -> Path:
    chain = result.structure.chain_id
    lines = [
        "# PyMOL session for comparative epitope mapping",
        f"# generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        f"load {result.structure.path.resolve()}, target",
        "hide everything",
        "show cartoon, target",
        f"show surface, target and chain {chain}",
        "set transparency, 0.25",
        "color grey80, target",
        "",
        "# composite score painted into the B-factor column (0-100)",
        "alter target, b=0.0",
    ]
    for residue in result.residues:
        if residue.ref_number is None:
            continue
        value = max(0.0, min(1.0, residue.composite)) * 100.0
        if value <= 0:
            continue
        lines.append(
            f"alter target and chain {residue.chain or chain} and resi "
            f"{residue.ref_number}, b={value:.2f}"
        )
    lines += [
        "rebuild",
        "spectrum b, white_red, target, minimum=0, maximum=100",
        "",
        "# top candidate patches",
    ]
    for index, patch in enumerate(result.patches[: result.config.top_n]):
        color = _PATCH_COLORS[index % len(_PATCH_COLORS)]
        selection = _pymol_selection(patch, chain)
        lines += [
            f"select {patch.patch_id}, {selection}",
            f"color {color}, {patch.patch_id}",
            f"show sticks, {patch.patch_id} and not (name C+N+O)",
            f"set surface_color, {color}, {patch.patch_id}",
        ]
    if result.patches:
        lines += [
            "",
            f"orient {result.patches[0].patch_id}",
            f"zoom {result.patches[0].patch_id}, 8",
        ]
    lines += [
        "deselect",
        "set ray_opaque_background, 0",
        "# residues.tsv holds the per-residue values behind these colours",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


# --------------------------------------------------------------------------
# report.md
# --------------------------------------------------------------------------


def _binding_table(result: RunResult) -> str:
    lines = ["| species | binding | identity to reference | role |", "|---|---|---|---|"]
    for record in result.dataset.records:
        identity = result.alignment.identities.get(record.name)
        identity_text = f"{identity:.1f}%" if identity is not None else "reference"
        role = "reference" if record.name == result.alignment.reference else ""
        lines.append(f"| {record.name} | {record.call} | {identity_text} | {role} |")
    return "\n".join(lines)


def _counts_table(result: RunResult) -> str:
    labels = {
        "all_reference_residues": "all reference residues",
        "in_ectodomain": "within the analysed range",
        "discriminating": f"discriminating (score >= {result.config.discrimination_cutoff})",
        "discriminating_and_exposed": f"and exposed (RSA >= {result.config.rsa_cutoff})",
        "after_context_masking": "and not masked by context/glycan/ectodomain filters",
    }
    lines = ["| narrowing step | candidates |", "|---|---|"]
    for key, label in labels.items():
        lines.append(f"| {label} | {result.counts[key]} |")
    in_patches = sum(p.size for p in result.patches)
    lines.append(f"| clustered into {len(result.patches)} patch(es) | {in_patches} |")
    if result.patches:
        top = result.patches[0]
        lines.append(f"| top-ranked patch {top.patch_id} | {top.size} |")
    return "\n".join(lines)


def _patch_section(result: RunResult) -> str:
    if not result.patches:
        return (
            "No patch met the current thresholds. Either the discriminating "
            "residues are scattered (lower `--discrimination-cutoff` or raise "
            "`--patch-radius` to see the near misses) or the structure has little "
            "exposed surface in the analysed range.\n"
        )
    blocks: List[str] = []
    for patch in result.patches[: result.config.top_n]:
        members = "\n".join(
            f"  - `{m.ref_number or m.ref_index + 1}` {m.aa} -> "
            + ", ".join(
                f"{name}:{m.species_residues.get(name, MISSING)}"
                for name in result.alignment.sequences
                if name != result.alignment.reference
            )
            + f" (discrimination {m.discrimination:.2f}, RSA {_fmt(m.rsa, 2)}"
            + (f", pLDDT {m.plddt:.0f}" if m.plddt == m.plddt else "")
            + (f", {m.plddt_flag}" if m.plddt_flag else "")
            + (", indel" if m.involves_gap else "")
            + (", glycan-proximal" if m.glycan_flags else "")
            + ")"
            for m in patch.members
        )
        chimera = next(
            (c for c in result.chimeras if c.patch_id == patch.patch_id), None
        )
        chimera_text = (
            f"- Suggested chimera swap: reference `{chimera.label}` "
            f"({len(chimera.segments)} segment(s), {chimera.length} residues total; "
            f"{chimera.note})\n"
            if chimera
            else ""
        )
        mutant_text = ""
        patch_mutants = [m for m in result.mutants if m.patch_id == patch.patch_id][:6]
        if patch_mutants:
            mutant_text = "- Priority mutants (each numbered in its own background): " + ", ".join(
                f"{m.background} {m.label} ({m.direction.replace('_', ' ')})"
                for m in patch_mutants
            ) + "\n"
        blocks.append(
            f"### {patch.patch_id} - rank {patch.rank_raw} raw, "
            f"{patch.rank_normalized} size-normalised\n\n"
            f"- Score: total {patch.total_score:.3f}, normalised "
            f"{patch.normalized_score:.3f}, mean {patch.mean_score:.3f}\n"
            f"- Geometry: {patch.size} residues, spread {patch.spread:.1f} A, "
            f"accessible area {patch.surface_area:.0f} A^2, mean RSA "
            f"{_fmt(patch.mean_rsa, 2)}\n"
            f"- Chemistry: max Grantham {patch.max_grantham:.0f}, "
            f"{patch.n_indels} indel position(s)\n"
            + (f"- Flags: {' | '.join(patch.flags)}\n" if patch.flags else "")
            + chimera_text
            + mutant_text
            + f"- Members:\n{members}\n"
        )
    return "\n".join(blocks)


def _degeneracy_section(result: RunResult) -> str:
    d = result.degeneracy
    background = (
        f"{d.background_fraction:.1%}" if d.background_fraction == d.background_fraction else "n/a"
    )
    enrichment = (
        "infinite (no random relabelling produced any discriminating column)"
        if math.isinf(d.enrichment)
        else f"{d.enrichment:.2f}x"
    )
    lines = [
        f"- Binding groups: {d.n_binders} binder(s) vs {d.n_non_binders} non-binder(s).",
        f"- {d.observed_fraction:.1%} of positions are discriminating at the current "
        f"cutoff ({d.cutoff}).",
        f"- Background from {'all' if d.exhaustive else d.n_labelings} alternative "
        f"binder/non-binder labellings of the same species: {background}.",
        f"- Enrichment of the real labelling over chance: {enrichment}.",
    ]
    if d.clade_split:
        lines.append(
            f"- **The binding pattern coincides with the sequence tree.** "
            f"{d.clade_split_detail}. Sequence discrimination alone therefore carries "
            "little information here; the ranking below rests on the structural "
            "filters, and the candidate list should be treated as weak."
        )
    else:
        lines.append(
            "- The binding pattern cuts across the sequence tree, which is the "
            "favourable case: discriminating positions are less likely to be "
            "neutral clade markers."
        )
    if d.enrichment < 1.5 and not math.isinf(d.enrichment):
        lines.append(
            "- Enrichment below ~1.5x means the observed labelling is barely better "
            "than a random one. Add species with mixed binding outcomes before "
            "committing to any candidate."
        )
    return "\n".join(lines)


def _glycan_section(result: RunResult) -> str:
    glycans = result.glycans
    if not glycans.differential:
        return (
            f"{len(glycans.sequons)} N-X-S/T sequon(s) found across the scored "
            "species; none separates binders from non-binders cleanly.\n"
        )
    lines = [
        f"{len(glycans.differential)} differential sequon(s) separate the two groups:",
        "",
        "| reference position | motif | present in | absent in | residues flagged nearby |",
        "|---|---|---|---|---|",
    ]
    for sequon in glycans.differential:
        nearby = sum(
            1
            for residue in result.residues
            if any(
                sequon.ref_number and sequon.ref_number in flag
                for flag in residue.glycan_flags
            )
        )
        lines.append(
            f"| {sequon.ref_number or 'unmodelled'} | {sequon.motif} | "
            f"{', '.join(sequon.present_in)} | {', '.join(sequon.absent_in)} | {nearby} |"
        )
    lines.append("")
    lines.append(
        f"A glycan at any of these sites reaches roughly {glycans.radius:.0f} A, so the "
        "difference that abolishes binding may sit well outside the antibody "
        "footprint. Treat the flagged neighbourhood as one hypothesis, not as part "
        "of the epitope."
    )
    return "\n".join(lines)


def _next_experiments(result: RunResult) -> str:
    if not result.patches:
        return "- No patch reached threshold, so no experiments are proposed yet.\n"
    lines = [
        "1. **Chimeras first.** Swap the segment(s) below from the non-binder into "
        "the binder background (and back) to localise the determinant before "
        "making point mutants:",
    ]
    for chimera in result.chimeras:
        lines.append(
            f"   - {chimera.patch_id}: reference `{chimera.label}` "
            f"({chimera.length} residues over {len(chimera.segments)} segment(s), "
            f"{chimera.n_discriminating_included} discriminating position(s) inside"
            + (
                f", also contains {', '.join(chimera.other_patches_included)}"
                if chimera.other_patches_included
                else ""
            )
            + f"). {chimera.note}."
        )
    gain = [m for m in result.mutants if m.direction == "gain_of_binding"][:8]
    loss = [m for m in result.mutants if m.direction == "loss_of_binding"][:8]
    lines.append(
        "2. **Gain-of-binding mutants are the convincing experiment.** Introduce "
        "binder residues into the non-binder background:"
    )
    for m in gain:
        lines.append(
            f"   - {m.background} {m.label} [{m.numbering}] "
            f"(reference position {m.ref_number}, Grantham {m.grantham:.0f}, "
            f"RSA {_fmt(m.rsa, 2)})"
        )
    lines.append(
        "3. **Loss-of-binding mutants** in the binder background support the same "
        "hypothesis but can also reflect generic misfolding, so read them only "
        "alongside the reciprocal:"
    )
    for m in loss:
        lines.append(
            f"   - {m.background} {m.label} [{m.numbering}] "
            f"(Grantham {m.grantham:.0f}, RSA {_fmt(m.rsa, 2)})"
        )
    lines.append(
        "4. **Add species.** Any additional ortholog with a known binding outcome - "
        "especially one that breaks the current clade split - cuts the candidate "
        "list faster than any of the above."
    )
    return "\n".join(lines)


def write_report(result: RunResult, path: Path) -> Path:
    config = result.config
    structure = result.structure
    singletons = result.singletons

    parameters = "\n".join(
        f"| {key} | {value} |" for key, value in sorted(config.as_dict().items())
    )
    warnings_block = (
        "\n".join(f"- {w}" for w in result.warnings)
        if result.warnings
        else "- none"
    )
    singleton_block = (
        "\n".join(
            f"- `{p.members[0].ref_number}` {p.members[0].aa} "
            f"(composite {p.members[0].composite:.3f}, discrimination "
            f"{p.members[0].discrimination:.2f}, RSA {_fmt(p.members[0].rsa, 2)})"
            for p in singletons[:20]
        )
        if singletons
        else "- none"
    )

    plddt_note = ""
    if structure.is_alphafold:
        low = sum(1 for r in result.residues if r.plddt == r.plddt and r.plddt < 70)
        unreliable = sum(1 for r in result.residues if r.plddt == r.plddt and r.plddt < 50)
        plddt_note = (
            f"\nThe structure is an AlphaFold model: {low} residue(s) below pLDDT 70 "
            f"and {unreliable} below 50 are down-weighted but kept, since flexible "
            "loops are exactly where epitopes often sit.\n"
        )

    text = f"""# Comparative epitope mapping report

Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by
epitope-map {result.version}.

**This is a ranked hypothesis list to guide chimera and point-mutant design. It
is not a prediction of the epitope.**

## 1. Run parameters

| parameter | value |
|---|---|
{parameters}

Structure: `{structure.path.name}`, chain `{structure.chain_id}`,
{len(structure.residues)} modelled residues, SASA backend
`{structure.sasa_backend}`, DSSP {'used' if structure.dssp_used else 'unavailable'}.
{plddt_note}
## 2. Species and alignment

{_binding_table(result)}

Alignment method: `{result.alignment.method}`, {result.alignment.length} columns,
{len(result.residue_map)} reference residues.

## 3. How much signal is there?

{_degeneracy_section(result)}

## 4. Narrowing

{_counts_table(result)}

The alignment does almost none of the narrowing in a two-clade comparison; the
structural filters do. Read the table above as the honest candidate count at
each step.

## 5. Top candidate patches

{_patch_section(result)}

### Singletons (low priority)

Isolated discriminating surface residues that did not cluster. They are reported
rather than dropped, but a single residue rarely accounts for a species-specific
loss of binding on its own.

{singleton_block}

## 6. Glycosylation

{_glycan_section(result)}

## 7. Suggested next experiments

{_next_experiments(result)}

## 8. Warnings from this run

{warnings_block}

## 9. Caveats that apply to every run

{chr(10).join(f'{i}. {c}' for i, c in enumerate(CAVEATS, start=1))}

For calibration: a typical conformational epitope is
{TYPICAL_EPITOPE_RESIDUES[0]}-{TYPICAL_EPITOPE_RESIDUES[1]} residues burying
{TYPICAL_EPITOPE_BSA[0]:.0f}-{TYPICAL_EPITOPE_BSA[1]:.0f} A^2, of which only 4-6
are energetic hot spots. Patches far outside that envelope are flagged in
`patches.tsv`.

## 10. Files

- `residues.tsv` - every reference residue with all scoring factors kept separate
- `patches.tsv` - ranked patches (raw and size-normalised), including singletons
- `chimeras.tsv` - suggested domain-swap segments, one row per segment
- `mutants.tsv` - reciprocal point-mutant suggestions in both directions
- `alignment.fasta` - the MSA actually used
- `session.pml` - PyMOL session: composite score painted white to red, top
  patches coloured
- `report.md` - this file
"""
    path.write_text(text)
    return path
