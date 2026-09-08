"""Validation harness: how far down the ranked list is the true epitope?

Metrics reported per case (spec section 6):

* is any true contact residue inside the top-ranked patch? the top 3?
* what fraction of true contact residues is recovered by those patches?
* how many candidates must be tested to reach them, versus the naive baseline
  of "every exposed discriminating residue"?

Run it on the built-in synthetic case with::

    python validation/benchmark.py synthetic

or on a real case defined in YAML (see ``validation/cases/``)::

    python validation/benchmark.py validation/cases/d1.3_hel.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from epitope_map.pipeline import RunConfig, RunResult, run_pipeline  # noqa: E402
from epitope_map.report import write_all  # noqa: E402
from epitope_map.structure import distance  # noqa: E402


@dataclass
class BenchmarkMetrics:
    case: str
    truth: List[str]
    n_reference_residues: int
    naive_baseline: int
    top_patch_size: int
    hit_in_top1: bool
    hit_in_top3: bool
    recall_top1: float
    recall_top3: float
    candidates_to_first_hit: Optional[int]
    enrichment_over_baseline: float
    degenerate: bool
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {k: v for k, v in self.__dict__.items()}

    def render(self) -> str:
        lines = [
            f"case: {self.case}",
            f"  true contact residues:          {len(self.truth)} "
            f"({', '.join(self.truth[:12])}{' ...' if len(self.truth) > 12 else ''})",
            f"  reference residues analysed:    {self.n_reference_residues}",
            f"  naive baseline (exposed+disc.): {self.naive_baseline}",
            f"  top patch size:                 {self.top_patch_size}",
            f"  true epitope hit in top 1 / 3:  {self.hit_in_top1} / {self.hit_in_top3}",
            f"  recall in top 1 / top 3:        {self.recall_top1:.0%} / {self.recall_top3:.0%}",
            f"  candidates tested to 1st hit:   {self.candidates_to_first_hit}",
            f"  enrichment over baseline:       {self.enrichment_over_baseline:.2f}x",
            f"  flagged as degenerate:          {self.degenerate}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def contact_residues(
    structure_spec: str,
    antigen_chain: str,
    partner_chains: Sequence[str],
    cutoff: float = 4.5,
    cache_dir: Optional[Path] = None,
) -> List[str]:
    """True epitope from a solved complex: antigen residues near the antibody.

    Objective ground truth - it comes from the complex itself, not from anyone's
    recollection of which residues matter.
    """
    from epitope_map.structure import _open_structure, resolve_structure

    path = resolve_structure(structure_spec, cache_dir=cache_dir)
    structure, _ = _open_structure(path)
    model = next(iter(structure))
    antigen = model[antigen_chain]
    partner_atoms = [
        atom
        for chain_id in partner_chains
        for atom in model[chain_id].get_atoms()
        if atom.element != "H"
    ]
    contacts: List[str] = []
    for residue in antigen:
        if residue.get_id()[0].strip() and residue.get_resname().strip() != "MSE":
            continue
        for atom in residue:
            if atom.element == "H":
                continue
            if any(distance(atom.coord, other.coord) <= cutoff for other in partner_atoms):
                het, resseq, icode = residue.get_id()
                code = icode.strip()
                contacts.append(f"{resseq}{code}" if code else str(resseq))
                break
    return contacts


def evaluate(result: RunResult, truth: Sequence[str], case: str) -> BenchmarkMetrics:
    truth_set: Set[str] = set(truth)
    patches = result.patches
    top1 = {m.ref_number for m in patches[0].members} if patches else set()
    top3: Set[str] = set()
    for patch in patches[:3]:
        top3 |= {m.ref_number for m in patch.members}

    # how many residues must be tested, walking the ranked list, before the
    # first true contact residue is reached
    tested = 0
    first_hit: Optional[int] = None
    for patch in patches:
        for member in patch.members:
            tested += 1
            if member.ref_number in truth_set and first_hit is None:
                first_hit = tested
    baseline = result.counts["discriminating_and_exposed"]
    enrichment = (
        (len(truth_set & top1) / max(1, len(top1)))
        / max(1e-9, len(truth_set) / max(1, baseline))
        if top1
        else 0.0
    )
    notes: List[str] = []
    if result.degeneracy.clade_split:
        notes.append(
            "binders and non-binders coincide with the sequence tree - the run is "
            "expected to be weak, and the tool says so"
        )
    if not patches:
        notes.append("no patch reached threshold")
    return BenchmarkMetrics(
        case=case,
        truth=sorted(truth_set, key=lambda x: (len(x), x)),
        n_reference_residues=len(result.residues),
        naive_baseline=baseline,
        top_patch_size=len(top1),
        hit_in_top1=bool(truth_set & top1),
        hit_in_top3=bool(truth_set & top3),
        recall_top1=len(truth_set & top1) / len(truth_set) if truth_set else 0.0,
        recall_top3=len(truth_set & top3) / len(truth_set) if truth_set else 0.0,
        candidates_to_first_hit=first_hit,
        enrichment_over_baseline=enrichment,
        degenerate=result.degeneracy.is_degenerate,
        notes=notes,
    )


def run_synthetic(outdir: Path) -> BenchmarkMetrics:
    import synthetic

    inputs = synthetic.write_inputs(outdir / "inputs")
    config = RunConfig(
        sequences=str(inputs["sequences"]),
        binding=str(inputs["binding"]),
        reference="mouse",
        structure=str(inputs["structure"]),
        outdir=outdir / "results",
    )
    result = run_pipeline(config)
    write_all(result, config.outdir)
    return evaluate(result, synthetic.truth_numbers(), "synthetic (planted epitope)")


def run_scaling(outdir: Path) -> List[Dict[str, object]]:
    """What does one more informative species actually buy?

    Runs the synthetic case twice: once as the degenerate two-clade panel, once
    with an extra rodent that carries the planted substitutions (so the binding
    pattern no longer follows the sequence tree).
    """
    import synthetic

    rows: List[Dict[str, object]] = []
    for label, informative in (("two clades", False), ("+1 informative species", True)):
        directory = outdir / label.replace(" ", "_").replace("+", "plus")
        inputs = synthetic.write_inputs(directory / "inputs", include_informative=informative)
        config = RunConfig(
            sequences=str(inputs["sequences"]),
            binding=str(inputs["binding"]),
            reference="mouse",
            structure=str(inputs["structure"]),
            outdir=directory / "results",
        )
        result = run_pipeline(config)
        write_all(result, config.outdir)
        metrics = evaluate(result, synthetic.truth_numbers(), label)
        rows.append(
            {
                "panel": label,
                "species": len(result.dataset.records),
                "candidates": result.counts["discriminating_and_exposed"],
                "top_patch": metrics.top_patch_size,
                "recall_top1": round(metrics.recall_top1, 3),
                "clade_split": result.degeneracy.clade_split,
                "enrichment_over_chance": (
                    round(result.degeneracy.enrichment, 2)
                    if result.degeneracy.enrichment == result.degeneracy.enrichment
                    else None
                ),
            }
        )
    return rows


def run_case(case_path: Path, outdir: Path) -> BenchmarkMetrics:
    import yaml

    case = yaml.safe_load(case_path.read_text())
    base = case_path.parent
    def resolve(value: str) -> str:
        candidate = base / str(value)
        return str(candidate) if candidate.exists() else str(value)

    config = RunConfig(
        sequences=resolve(case["sequences"]),
        binding=resolve(case["binding"]),
        reference=str(case["reference"]),
        structure=resolve(case["structure"]),
        chain=case.get("chain"),
        outdir=outdir / "results",
        ectodomain=tuple(case["ectodomain"]) if case.get("ectodomain") else None,
        cache_dir=Path(case.get("cache_dir", outdir / "cache")),
    )
    result = run_pipeline(config)
    write_all(result, config.outdir)

    truth = case.get("true_epitope")
    if not truth:
        truth = contact_residues(
            resolve(case["complex_structure"]),
            case["complex_antigen_chain"],
            case["complex_partner_chains"],
            cutoff=float(case.get("contact_cutoff", 4.5)),
            cache_dir=config.cache_dir,
        )
    metrics = evaluate(result, [str(t) for t in truth], case.get("name", case_path.stem))
    if case.get("notes"):
        metrics.notes.append(str(case["notes"]))
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", help="'synthetic', 'scaling', or a path to a case YAML file"
    )
    parser.add_argument("--outdir", type=Path, default=Path("validation/out"))
    parser.add_argument("--json", type=Path, help="also write metrics as JSON")
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.case == "scaling":
        rows = run_scaling(args.outdir / "scaling")
        header = list(rows[0])
        widths = [max(len(h), *(len(str(r[h])) for r in rows)) for h in header]
        print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
        for row in rows:
            print("  ".join(str(row[h]).ljust(w) for h, w in zip(header, widths)))
        if args.json:
            args.json.write_text(json.dumps(rows, indent=2))
        return 0
    if args.case == "synthetic":
        metrics = run_synthetic(args.outdir / "synthetic")
    else:
        metrics = run_case(Path(args.case), args.outdir / Path(args.case).stem)
    print(metrics.render())
    if args.json:
        args.json.write_text(json.dumps(metrics.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
