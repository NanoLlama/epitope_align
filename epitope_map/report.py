"""Output writers: TSVs, report.md, alignment.fasta and the PyMOL session."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from .patches import (
    EPITOPE_SCALE,
    TYPICAL_EPITOPE_BSA,
    TYPICAL_EPITOPE_RESIDUES,
    Patch,
)
from .pipeline import RunResult
from .score import MIN_SPECIES_FOR_ENRICHMENT, MISSING

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
        "regions": write_regions(result, outdir / "regions.tsv"),
        "chimeras": write_chimeras(result, outdir / "chimeras.tsv"),
        "divergence": write_divergence_plot(result, outdir / "divergence.svg"),
        "patches": write_patches(result, outdir / "patches.tsv"),
        "mutants": write_mutants(result, outdir / "mutants.tsv"),
        "alignment": write_alignment(result, outdir / "alignment.fasta"),
        "pymol": write_pymol(result, outdir / "session.pml"),
        "report": write_report(result, outdir / "report.md"),
    }
    if result.radius_sensitivity:
        paths["radius_sensitivity"] = write_radius_sensitivity(
            result, outdir / "radius_sensitivity.tsv"
        )
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
                "topology": residue.topology,
                "domain": residue.domain,
                "accessible": residue.accessible,
                "discrimination": round(residue.discrimination, 4),
                "pattern_consistency": round(residue.pattern_consistency, 4),
                "max_grantham": round(residue.max_grantham, 1),
                "conserved_in_binders": residue.conserved_in_binders,
                "indel": residue.involves_gap,
                "indel_length": residue.indel_length,
                "alignment_confidence": _fmt(residue.alignment_confidence, 3),
                "local_identity": _fmt(residue.local_identity, 1),
                "low_identity_window": residue.low_identity_window,
                "missing_species": residue.has_missing,
                "sasa": _fmt(residue.sasa, 1),
                "rsa": _fmt(residue.rsa, 3),
                "buried": residue.buried,
                "plddt": _fmt(residue.plddt, 1),
                "plddt_flag": residue.plddt_flag,
                "disordered_region": residue.in_disordered_region,
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
                "n_discriminating": patch.n_discriminating,
                "n_total_surface": patch.n_total_surface,
                "residues": ",".join(patch.residue_labels),
                "context_residues": ",".join(
                    f"{r.aa}{r.ref_number or r.ref_index + 1}" for r in patch.context
                ),
                "neighbours": ",".join(
                    f"{pid}@{gap:.0f}A" for pid, gap in patch.neighbours
                ),
                "ref_numbers": ",".join(
                    m.ref_number or str(m.ref_index + 1) for m in patch.members
                ),
                "total_score": round(patch.total_score, 4),
                "mean_score": round(patch.mean_score, 4),
                "normalized_score": round(patch.normalized_score, 4),
                "max_grantham": round(patch.max_grantham, 1),
                "mean_rsa": _fmt(patch.mean_rsa, 3),
                "accessible_area_A2": _fmt(patch.surface_area, 1),
                "accessible_area_with_context_A2": _fmt(
                    patch.surface_area_with_context, 1
                ),
                "spread_A": _fmt(patch.spread, 1),
                "n_indels": patch.n_indels,
                "n_glycan_flagged": patch.n_glycan_flagged,
                "n_low_confidence_alignment": sum(
                    1
                    for m in patch.members
                    if (
                        m.alignment_confidence == m.alignment_confidence
                        and m.alignment_confidence < 0.7
                    )
                    or m.low_identity_window
                ),
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
    for label, group in (
        ("promoted", result.promoted_singletons),
        ("singleton", result.singletons),
    ):
        extra = patches_dataframe(result, group)
        if extra.empty:
            continue
        extra = extra.copy()
        extra["patch_id"] = extra["patch_id"].apply(lambda p: f"{p}({label})")
        frame = pd.concat([frame, extra], ignore_index=True)
    frame.to_csv(path, sep="\t", index=False)
    return path


def write_regions(result: RunResult, path: Path) -> Path:
    """Per-region enrichment: where in the chain the discrimination sits."""
    pd.DataFrame(region_rows(result)).to_csv(path, sep="\t", index=False)
    return path


def region_rows(result: RunResult) -> List[Dict[str, object]]:
    """One row per annotated region (domain, topology segment, or disorder).

    The stalk artifact in the first real run showed 34% of positions
    discriminating against a ~9% background, and nothing in the output said so.
    This is the table that says so.
    """
    regions: List[Tuple[str, str, range]] = []
    for segment in result.domain_segments:
        regions.append(
            (segment.description or "domain", "domain", range(segment.start, segment.end + 1))
        )
    for segment in result.topology.segments:
        regions.append((segment.label, "topology", range(segment.start, segment.end + 1)))
    for start, end, mean, count in result.structure.disordered_regions:
        first = result.residue_map.ref_index_of_key(start)
        last = result.residue_map.ref_index_of_key(end)
        if first is None or last is None:
            continue
        regions.append(
            (
                f"disordered {start.label}-{end.label} (pLDDT {mean:.0f})",
                "disorder",
                range(first + 1, last + 2),
            )
        )
    # always carry the whole-chain baseline: a region is only a hotspot relative
    # to the rest of the protein, and "34% here" means nothing without "9% there"
    regions.append(("whole chain (baseline)", "all", range(1, len(result.residues) + 1)))

    cutoff = result.config.discrimination_cutoff
    rows: List[Dict[str, object]] = []
    for name, kind, span in regions:
        members = [r for r in result.residues if r.ref_index + 1 in span]
        if not members:
            continue
        discriminating = [r for r in members if r.discrimination >= cutoff]
        exposed = [r for r in discriminating if not r.buried and r.accessible]
        plddt = [r.plddt for r in members if r.plddt == r.plddt]
        rsa = [r.rsa for r in members if r.rsa == r.rsa]
        rows.append(
            {
                "region": name,
                "kind": kind,
                "start": span.start,
                "end": span.stop - 1,
                "n_residues": len(members),
                "percent_discriminating": round(
                    100.0 * len(discriminating) / len(members), 1
                ),
                "percent_discriminating_and_exposed": round(
                    100.0 * len(exposed) / len(members), 1
                ),
                "mean_plddt": round(sum(plddt) / len(plddt), 1) if plddt else "",
                "mean_rsa": round(sum(rsa) / len(rsa), 3) if rsa else "",
                "percent_reachable": round(
                    100.0 * sum(1 for r in members if r.accessible) / len(members), 1
                ),
            }
        )
    baseline = [row for row in rows if row["kind"] == "all"]
    rest = [row for row in rows if row["kind"] != "all"]
    rest.sort(key=lambda row: -float(row["percent_discriminating"]))
    return rest + baseline


def write_radius_sensitivity(result: RunResult, path: Path) -> Path:
    pd.DataFrame(result.radius_sensitivity).to_csv(path, sep="\t", index=False)
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
            "verified_equivalence": m.verified,
            "caveat": m.caveat,
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
                    "constructible": chimera.constructible,
                    "problems": " | ".join(chimera.problems),
                    "domain_swap": chimera.domain_swap,
                    "patch_total_swapped_residues": chimera.length,
                    "other_patches_included": ",".join(chimera.other_patches_included),
                    "discriminating_positions_inside": chimera.n_discriminating_included,
                    "note": chimera.note,
                }
            )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


def write_divergence_plot(result: RunResult, path: Path) -> Path:
    """A windowed discrimination profile as hand-written SVG (no plotting deps).

    A spike over a disordered stalk and a broad elevation over the real epitope
    look identical in a table and completely different in a picture.
    """
    residues = result.residues
    if not residues:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
        return path

    width, height = 900, 320
    left, right, top, bottom = 60, 20, 30, 60
    plot_w = width - left - right
    plot_h = height - top - bottom
    n = len(residues)
    window = max(3, n // 60)

    smoothed = []
    for index in range(n):
        low = max(0, index - window)
        high = min(n, index + window + 1)
        chunk = [residues[i].discrimination for i in range(low, high)]
        smoothed.append(sum(chunk) / len(chunk))
    peak = max(smoothed) or 1.0

    def x_of(index: int) -> float:
        return left + plot_w * index / max(1, n - 1)

    def y_of(value: float) -> float:
        return top + plot_h * (1.0 - value / peak)

    bands = []
    for start, end, mean, count in result.structure.disordered_regions:
        first = result.residue_map.ref_index_of_key(start)
        last = result.residue_map.ref_index_of_key(end)
        if first is None or last is None:
            continue
        bands.append((first, last, "disordered", "#d94f4f"))
    for segment in result.topology.segments:
        if segment.kind == "extracellular":
            continue
        bands.append((segment.start - 1, segment.end - 1, segment.kind, "#8a8a8a"))
    for segment in result.domain_segments:
        bands.append(
            (segment.start - 1, segment.end - 1, segment.description or "domain", "#4f7fd9")
        )

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' font-family='system-ui, sans-serif'>",
        f"<rect width='{width}' height='{height}' fill='white'/>",
        f"<text x='{left}' y='18' font-size='13' fill='#222'>Discrimination along "
        f"the reference chain ({result.alignment.reference}), "
        f"{2 * window + 1}-residue window</text>",
    ]
    for start, end, label, colour in bands:
        x0, x1 = x_of(max(0, start)), x_of(min(n - 1, end))
        parts.append(
            f"<rect x='{x0:.1f}' y='{top}' width='{max(1.0, x1 - x0):.1f}' "
            f"height='{plot_h}' fill='{colour}' opacity='0.12'/>"
        )
        parts.append(
            f"<text x='{x0 + 2:.1f}' y='{top + 12}' font-size='9' fill='{colour}'>"
            f"{_escape(label)}</text>"
        )

    cutoff_y = y_of(min(peak, result.config.discrimination_cutoff))
    parts.append(
        f"<line x1='{left}' y1='{cutoff_y:.1f}' x2='{width - right}' "
        f"y2='{cutoff_y:.1f}' stroke='#c00' stroke-dasharray='4 3' stroke-width='1'/>"
    )
    parts.append(
        f"<text x='{width - right - 4}' y='{cutoff_y - 4:.1f}' font-size='9' "
        f"fill='#c00' text-anchor='end'>cutoff {result.config.discrimination_cutoff}</text>"
    )

    points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(smoothed))
    parts.append(
        f"<polyline points='{points}' fill='none' stroke='#1a1a1a' stroke-width='1.5'/>"
    )

    for patch in result.patches[: result.config.top_n]:
        for member in patch.members:
            parts.append(
                f"<circle cx='{x_of(member.ref_index):.1f}' "
                f"cy='{y_of(smoothed[member.ref_index]):.1f}' r='3' fill='#1a7f37'/>"
            )

    parts.append(
        f"<line x1='{left}' y1='{top + plot_h}' x2='{width - right}' "
        f"y2='{top + plot_h}' stroke='#444'/>"
    )
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#444'/>")
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        index = int(fraction * (n - 1))
        label = residues[index].ref_number or str(index + 1)
        parts.append(
            f"<text x='{x_of(index):.1f}' y='{top + plot_h + 16}' font-size='10' "
            f"fill='#444' text-anchor='middle'>{_escape(label)}</text>"
        )
    parts.append(
        f"<text x='{left + plot_w / 2:.0f}' y='{height - 22}' font-size='11' "
        f"fill='#444' text-anchor='middle'>reference residue number</text>"
    )
    for fraction in (0.0, 0.5, 1.0):
        value = peak * fraction
        parts.append(
            f"<text x='{left - 6}' y='{y_of(value) + 3:.1f}' font-size='10' "
            f"fill='#444' text-anchor='end'>{value:.2f}</text>"
        )
    parts.append(
        f"<text x='16' y='{top + plot_h / 2:.0f}' font-size='11' fill='#444' "
        f"transform='rotate(-90 16 {top + plot_h / 2:.0f})' text-anchor='middle'>"
        "discrimination</text>"
    )
    parts.append(
        f"<text x='{left}' y='{height - 6}' font-size='9' fill='#666'>"
        "green = members of the top patches; red band = disordered region; "
        "grey = not extracellular; blue = annotated domain</text>"
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")
    return path


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("'", "&apos;")
    )


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
            + (f", indel of {m.indel_length}" if m.involves_gap else "")
            + (
                f", alignment confidence {m.alignment_confidence:.2f}"
                if m.alignment_confidence == m.alignment_confidence
                and m.alignment_confidence < 0.7
                else ""
            )
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
            f"{chimera.note})"
            + (
                f"\n- Whole-domain alternative: swap `{chimera.domain_swap}`"
                if chimera.domain_swap
                else ""
            )
            + (
                "\n- NOT CONSTRUCTIBLE as proposed: " + "; ".join(chimera.problems)
                if not chimera.constructible
                else ""
            )
            + "\n"
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


def _topology_section(result: RunResult) -> str:
    topology = result.topology
    lines = [f"- Topology: {topology.summary()}."]
    if topology.known and not topology.whole_chain:
        excluded = [r for r in result.residues if not r.accessible]
        lines.append(
            f"- {len(excluded)} residue(s) an antibody cannot reach were excluded "
            "from scoring entirely, not merely down-weighted."
        )
    elif topology.whole_chain:
        lines.append(
            "- The whole chain was declared accessible. If this is a membrane "
            "protein, that is wrong and the ranking below will contain residues "
            "no antibody can reach."
        )
    if result.structure.disordered_regions:
        described = ", ".join(
            f"{a.label}-{b.label} ({n} residues, mean pLDDT {mean:.0f})"
            for a, b, mean, n in result.structure.disordered_regions
        )
        lines.append(
            f"- Disordered region(s) excluded from patch seeding: {described}. "
            "AlphaFold models these as extended tethers, so their RSA is not a "
            "measure of exposure."
        )
    if result.structure.is_alphafold:
        lines.append(
            f"- Structure is an AlphaFold model of the monomer ({result.structure.assembly}); "
            "if the real receptor is an oligomer, its interface is reported here "
            "as exposed. Supply --assembly-context or --context-chains to correct that."
        )
    return "\n".join(lines)


def _regions_section(result: RunResult) -> str:
    rows = region_rows(result)
    if not rows:
        return "No annotated regions.\n"
    lines = [
        "| region | kind | residues | % discriminating | % disc. & exposed | mean pLDDT | % reachable |",
        "|---|---|---|---|---|---|---|",
    ]
    shown = [row for row in rows if row["kind"] != "all"][:13]
    shown += [row for row in rows if row["kind"] == "all"]
    for row in shown:
        lines.append(
            f"| {row['region']} | {row['kind']} | {row['start']}-{row['end']} "
            f"({row['n_residues']}) | {row['percent_discriminating']}% | "
            f"{row['percent_discriminating_and_exposed']}% | {row['mean_plddt']} | "
            f"{row['percent_reachable']}% |"
        )
    lines.append("")
    lines.append(
        "A region well above the rest of the chain is either the answer or an "
        "artifact, and the two look identical in a residue table. The least "
        "constrained part of a protein - a stalk, a linker - is discriminating "
        "everywhere without being an epitope anywhere."
    )
    return "\n".join(lines)


def _merged_surfaces_section(result: RunResult) -> str:
    if not result.merged_surfaces:
        return (
            "No two patches sit close enough to be one antibody footprint, so "
            "each is an independent hypothesis.\n"
        )
    lines = [
        "Patches whose nearest members are within one antibody footprint "
        f"(<= {EPITOPE_SCALE:.0f} A) are probably one surface split by the "
        "clustering radius. Compare the combined area, not each fragment's, "
        f"against the {TYPICAL_EPITOPE_BSA[0]:.0f}-{TYPICAL_EPITOPE_BSA[1]:.0f} "
        "A^2 an antibody buries.",
        "",
        "| patches | residues | combined area (A^2) | spread (A) | widest gap (A) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for surface in result.merged_surfaces[:5]:
        lines.append(
            f"| {' + '.join(surface['patches'])} | {surface['n_residues']} "
            f"({', '.join(str(r) for r in surface['residues'])}) | "
            f"{float(surface['accessible_area_A2']):.0f} | "
            f"{float(surface['spread_A']):.1f} | {float(surface['max_gap_A']):.1f} | "
            f"{surface['verdict']} |"
        )
    if result.radius_sensitivity:
        lines.append("")
        lines.append(
            "`radius_sensitivity.tsv` shows which of these merge at which "
            "clustering radius: patches that merge early are one surface, "
            "patches still separate at 18 A are distinct hypotheses."
        )
    return "\n".join(lines)


def _promoted_section(result: RunResult) -> str:
    if not result.promoted_singletons:
        return "None.\n"
    lines = [
        "Isolated residues that did not cluster but are worth attention anyway - "
        "being alone only means no *other above-cutoff* residue sat within the "
        "clustering radius, which is not evidence of being unimportant:",
        "",
    ]
    for patch in result.promoted_singletons[:10]:
        member = patch.members[0]
        lines.append(
            f"- `{member.ref_number}` {member.aa} (composite {member.composite:.2f}, "
            f"RSA {_fmt(member.rsa, 2)}) - {patch.promoted_reason}"
        )
    return "\n".join(lines)


def _equivalence_section(result: RunResult) -> str:
    if not result.equivalences:
        return ""
    lines = [
        "",
        "### Residue equivalences from structure",
        "",
    ]
    for species, equivalence in result.equivalences.items():
        lines.append(
            f"- {species}: superposed over {equivalence.n_superposed} residues at "
            f"{equivalence.rmsd:.2f} A RMSD; "
            + (
                f"the alignment names a different residue at "
                f"{len(equivalence.disagreements)} position(s), where the "
                "structural pairing was used instead"
                if equivalence.disagreements
                else "the alignment agrees everywhere it could be checked"
            )
        )
    return "\n".join(lines)


def _advice_section(result: RunResult) -> str:
    if not result.panel_advice:
        return "No panel advice could be computed.\n"
    lines = [
        "Which ortholog to test next, ranked by how much of the candidate list it "
        "would resolve. The hypothetical relative is modelled as sequence-identical "
        "to the species it sits beside, so these are **best cases** - read the "
        "ranking, not the numbers.",
        "",
        "| test next | candidates now | best case after | resolved |",
        "|---|---|---|---|",
    ]
    for entry in result.panel_advice[:6]:
        lines.append(
            f"| {entry['hypothetical_species']} | {entry['candidates_now']} | "
            f"{entry['candidates_after_best_case']} | "
            f"{float(entry['reduction_fraction']):.0%} |"
        )
    return "\n".join(lines)


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
        f"- {d.observed_fraction:.1%} of reachable positions are discriminating at "
        f"the current cutoff ({d.cutoff}).",
    ]
    if d.enrichment_is_meaningful:
        lines += [
            f"- Background from {d.n_labelings} alternative binder/non-binder "
            f"labellings of the same species{' (all of them)' if d.exhaustive else ''}: "
            f"{background}.",
            f"- Enrichment of the real labelling over chance: {enrichment}.",
        ]
    else:
        lines.append(
            f"- **No enrichment figure is quoted.** With {d.n_scored} scored species "
            f"there are only {d.n_labelings} alternative labelling(s) to compare "
            "against, which is too few for the background rate to mean anything. "
            f"(For the record it came out at {background}.) At least "
            f"{MIN_SPECIES_FOR_ENRICHMENT} scored species are needed before this "
            "statistic is worth reading."
        )
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
    domain_swaps = sorted(
        {c.domain_swap for c in result.chimeras if c.domain_swap}
    )
    if domain_swaps:
        lines.append(
            "   - **Domain-level first:** the top patches sit in "
            + ", ".join(f"`{d}`" for d in domain_swaps)
            + ". One whole-domain swap distinguishes them in a single "
            "experiment and is far easier to build than the segment lists below."
        )
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
            + (
                "  **NOT CONSTRUCTIBLE AS PROPOSED**: "
                + "; ".join(chimera.problems)
                + "."
                if not chimera.constructible
                else ""
            )
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
            + ("  **UNVERIFIED** - " + m.caveat if not m.verified else "")
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
            + ("  **UNVERIFIED** - " + m.caveat if not m.verified else "")
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

## 1. Read this before the rankings

### What was excluded, and why

{_topology_section(result)}

### How much signal is there?

{_degeneracy_section(result)}

### Warnings from this run

{warnings_block}

### Where the divergence actually sits

{_regions_section(result)}

See `divergence.svg` for the same thing as a profile along the chain.

## 2. Run parameters

| parameter | value |
|---|---|
{parameters}

Structure: `{structure.path.name}`, chain `{structure.chain_id}`,
{len(structure.residues)} modelled residues, {structure.assembly}, SASA backend
`{structure.sasa_backend}`, DSSP {'used' if structure.dssp_used else 'unavailable'}.
{plddt_note}
## 3. Species and alignment

{_binding_table(result)}

Alignment method: `{result.alignment.method}`, {result.alignment.length} columns,
{len(result.residue_map)} reference residues.

## 4. Narrowing

{_counts_table(result)}

The alignment does almost none of the narrowing in a two-clade comparison; the
structural filters do. Read the table above as the honest candidate count at
each step.

## 5. Top candidate patches

{_patch_section(result)}

### Candidate merged surfaces

{_merged_surfaces_section(result)}

### High-priority isolated residues

{_promoted_section(result)}

### Remaining singletons (low priority)

Isolated discriminating surface residues with nothing to recommend them beyond
the score. Reported rather than dropped, but a single residue rarely accounts
for a species-specific loss of binding on its own.

{singleton_block}

## 6. Glycosylation

{_glycan_section(result)}

## 7. Suggested next experiments

{_next_experiments(result)}

### Which species to test next

{_advice_section(result)}
{_equivalence_section(result)}

## 8. Caveats that apply to every run

{chr(10).join(f'{i}. {c}' for i, c in enumerate(CAVEATS, start=1))}

For calibration: a typical conformational epitope is
{TYPICAL_EPITOPE_RESIDUES[0]}-{TYPICAL_EPITOPE_RESIDUES[1]} residues burying
{TYPICAL_EPITOPE_BSA[0]:.0f}-{TYPICAL_EPITOPE_BSA[1]:.0f} A^2, of which only 4-6
are energetic hot spots. Patches far outside that envelope are flagged in
`patches.tsv`.

## 9. Files

- `residues.tsv` - every reference residue with all scoring factors kept separate
- `regions.tsv` - per-region enrichment: where the divergence sits
- `patches.tsv` - ranked patches (raw and size-normalised), promoted isolated
  residues and singletons, with conserved surface context and neighbours
- `chimeras.tsv` - suggested domain-swap segments, one row per segment, with a
  constructibility verdict
- `mutants.tsv` - reciprocal point-mutant suggestions, marked where the residue
  equivalence behind them is not reliable
- `radius_sensitivity.tsv` - which patches merge at which clustering radius
  (written only with --radius-sweep)
- `divergence.svg` - discrimination along the chain, with domains, disordered
  regions and the top patches marked
- `alignment.fasta` - the MSA actually used
- `session.pml` - PyMOL session: composite score painted white to red, top
  patches coloured
- `report.md` - this file
"""
    path.write_text(text)
    return path
