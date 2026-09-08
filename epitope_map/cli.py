"""Command-line interface and YAML config loading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import __version__
from .align import AlignmentError
from .io_seq import InputError
from .pipeline import DEFAULT_RSA_CUTOFF, RunConfig, run_pipeline
from .report import write_all
from .score import DEFAULT_DISCRIMINATION_CUTOFF
from .glycan import DEFAULT_GLYCAN_RADIUS
from .patches import DEFAULT_MIN_PATCH_SIZE, DEFAULT_PATCH_RADIUS
from .structure import StructureError


def parse_range(text: str) -> Tuple[int, int]:
    """Parse ``start-end``, tolerating negative-free ranges only."""
    parts = str(text).replace("..", "-").split("-")
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected start-end, got {text!r}")
    try:
        start, end = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"range bounds must be integers: {text!r}")
    if start > end:
        raise argparse.ArgumentTypeError(f"range start after end: {text!r}")
    return start, end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epitope-map",
        description=(
            "Narrow down the likely epitope of a monoclonal antibody from "
            "cross-species binding data, sequence alignment and structural "
            "solvent accessibility. Produces a ranked hypothesis list, not a "
            "prediction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in worked example end to end and write its results "
        "to --outdir; use this to check the installation works before "
        "assembling your own inputs",
    )
    parser.add_argument("--config", type=Path, help="YAML config file with the same keys")
    parser.add_argument(
        "--sequences",
        help="FASTA path, or a comma-separated list of UniProt accessions "
        "(optionally 'label=ACCESSION')",
    )
    parser.add_argument("--binding", help="CSV/YAML file of species,binding calls")
    parser.add_argument("--reference", help="reference species; must be a binder")
    parser.add_argument(
        "--structure", help="PDB/mmCIF path, 4-character PDB ID, or AlphaFold accession"
    )
    parser.add_argument("--outdir", type=Path, default=Path("results"))
    parser.add_argument(
        "--ectodomain", type=parse_range, help="start-end range to analyse"
    )
    parser.add_argument(
        "--ectodomain-numbering",
        choices=("structure", "sequence"),
        default="structure",
        help="whether --ectodomain refers to structure author numbering or to "
        "1-based reference sequence positions",
    )
    parser.add_argument("--chain", help="chain ID (default: first protein chain)")
    parser.add_argument(
        "--assembly-context",
        help="second structure whose chains are added when computing SASA in context",
    )
    parser.add_argument(
        "--context-chains",
        default="",
        help="comma-separated chain IDs to include in the context SASA calculation",
    )
    parser.add_argument(
        "--occluding-structure",
        help="structure of a known interacting partner used to mask occluded residues",
    )
    parser.add_argument(
        "--occluding-chains",
        default="",
        help="comma-separated chains of the interacting partner",
    )
    parser.add_argument("--rsa-cutoff", type=float, default=DEFAULT_RSA_CUTOFF)
    parser.add_argument(
        "--discrimination-cutoff", type=float, default=DEFAULT_DISCRIMINATION_CUTOFF,
        help="minimum discrimination (normalized-Grantham units) to seed a patch",
    )
    parser.add_argument("--patch-radius", type=float, default=DEFAULT_PATCH_RADIUS)
    parser.add_argument("--min-patch-size", type=int, default=DEFAULT_MIN_PATCH_SIZE)
    parser.add_argument("--glycan-radius", type=float, default=DEFAULT_GLYCAN_RADIUS)
    parser.add_argument(
        "--cluster-method", choices=("graph", "dbscan"), default="graph"
    )
    parser.add_argument(
        "--aligner", choices=("auto", "mafft", "muscle", "pairwise"), default="auto"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument(
        "--mismatch-tolerance",
        type=float,
        default=0.05,
        help="maximum fraction of reference/structure residue mismatches before "
        "the run stops",
    )
    parser.add_argument(
        "--no-dssp", action="store_true", help="skip DSSP even if it is installed"
    )
    parser.add_argument("--cache-dir", type=Path, help="where fetched files are cached")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"epitope-map {__version__}")
    return parser


def _load_config_file(path: Path) -> Dict[str, object]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise InputError("PyYAML is required for --config") from exc
    data = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(data, dict):
        raise InputError("config file must contain a mapping of option names to values")
    return {str(k).replace("-", "_"): v for k, v in data.items()}


def _split_list(value) -> List[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def demo_config(outdir: Path) -> RunConfig:
    """Write the built-in example's input files and point a run at them."""
    from . import demo

    inputs = demo.write_inputs(Path(outdir) / "example-inputs")
    return RunConfig(
        sequences=str(inputs["sequences"]),
        binding=str(inputs["binding"]),
        reference="mouse",
        structure=str(inputs["structure"]),
        outdir=Path(outdir) / "results",
    )


def config_from_args(args: argparse.Namespace) -> RunConfig:
    values: Dict[str, object] = {}
    if args.config:
        values.update(_load_config_file(args.config))

    parser = build_parser()
    for key, value in vars(args).items():
        if key in ("config", "quiet", "demo"):
            continue
        if value is None:
            continue
        # command-line values win over the config file, but only when the user
        # actually passed them
        if key in values and value == parser.get_default(key):
            continue
        values[key] = value

    required = ("sequences", "binding", "reference", "structure")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise InputError(
            "missing required option(s): "
            + ", ".join(f"--{m.replace('_', '-')}" for m in missing)
        )

    ectodomain = values.get("ectodomain")
    if isinstance(ectodomain, str):
        ectodomain = parse_range(ectodomain)
    elif isinstance(ectodomain, (list, tuple)) and len(ectodomain) == 2:
        ectodomain = (int(ectodomain[0]), int(ectodomain[1]))

    return RunConfig(
        sequences=str(values["sequences"]),
        binding=str(values["binding"]),
        reference=str(values["reference"]),
        structure=str(values["structure"]),
        outdir=Path(values.get("outdir", "results")),
        ectodomain=ectodomain,
        ectodomain_numbering=str(values.get("ectodomain_numbering", "structure")),
        chain=str(values["chain"]) if values.get("chain") else None,
        assembly_context=str(values["assembly_context"]) if values.get("assembly_context") else None,
        context_chains=_split_list(values.get("context_chains")),
        occluding_chains=_split_list(values.get("occluding_chains")),
        occluding_structure=str(values["occluding_structure"]) if values.get("occluding_structure") else None,
        rsa_cutoff=float(values.get("rsa_cutoff", DEFAULT_RSA_CUTOFF)),
        discrimination_cutoff=float(
            values.get("discrimination_cutoff", DEFAULT_DISCRIMINATION_CUTOFF)
        ),
        patch_radius=float(values.get("patch_radius", DEFAULT_PATCH_RADIUS)),
        min_patch_size=int(values.get("min_patch_size", DEFAULT_MIN_PATCH_SIZE)),
        glycan_radius=float(values.get("glycan_radius", DEFAULT_GLYCAN_RADIUS)),
        cluster_method=str(values.get("cluster_method", "graph")),
        aligner=str(values.get("aligner", "auto")),
        threads=int(values.get("threads", 1)),
        top_n=int(values.get("top_n", 5)),
        mismatch_tolerance=float(values.get("mismatch_tolerance", 0.05)),
        run_dssp=not bool(values.get("no_dssp", False)),
        cache_dir=Path(values["cache_dir"]) if values.get("cache_dir") else None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.demo:
            outdir = args.outdir if args.outdir != Path("results") else Path("demo-run")
            config = demo_config(outdir)
        else:
            config = config_from_args(args)
        result = run_pipeline(config)
        paths = write_all(result, config.outdir)
    except (InputError, AlignmentError, StructureError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2

    if not args.quiet:
        if args.demo:
            print(
                "Demo run: six invented species (three that bind, three that do "
                "not) against a toy structure with a known answer.\n"
                "The eight residues of the planted epitope are S65, R66, T68, "
                "K70, D72, E114, R116 and E118 - the top patch below should be "
                "exactly those.\n"
                "The warnings are expected: they describe the honest limits of "
                "this toy example, and a real run prints its own.\n"
            )
        print(f"epitope-map {__version__}")
        print(
            f"{len(result.residues)} reference residues, "
            f"{result.counts['discriminating']} discriminating, "
            f"{result.counts['after_context_masking']} exposed and unmasked, "
            f"{len(result.patches)} patch(es)"
        )
        if result.degeneracy.clade_split:
            print(
                "WARNING: binders and non-binders coincide with the sequence tree; "
                "sequence discrimination carries little information in this run"
            )
        for warning in result.warnings:
            print(f"warning: {warning}")
        for patch in result.patches[: config.top_n]:
            print(
                f"  {patch.patch_id}: {patch.size} residues "
                f"({', '.join(patch.residue_labels)}) total {patch.total_score:.3f}"
            )
        for name, path in paths.items():
            print(f"wrote {path}")
        if args.demo:
            print(
                "\nInstallation looks healthy. Open "
                f"{paths['report']} to see the kind of write-up a real run "
                "produces, then follow the guide at "
                "https://github.com/NanoLlama/epitope_align/blob/main/"
                "GETTING_STARTED.md to assemble your own inputs."
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
