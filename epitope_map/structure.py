"""Structure parsing, SASA/RSA, pLDDT, secondary structure, geometry.

Everything in here is expressed in *author* numbering via :class:`ResidueKey`;
translation to alignment columns or reference sequence indices is the job of
:class:`epitope_map.align.ResidueMap` and happens nowhere else.
"""

from __future__ import annotations

import gzip
import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from Bio.PDB import MMCIFParser, PDBParser, ShrakeRupley
from Bio.PDB.Polypeptide import is_aa

from .data.max_asa import relative_sasa

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    # common modified residues, mapped to their parent
    "MSE": "M", "SEP": "S", "TPO": "T", "PTR": "Y", "CSO": "C", "HYP": "P",
    "PCA": "Q", "MLY": "K", "KCX": "K", "LLP": "K", "CME": "C", "CSD": "C",
}

_BACKBONE = {"N", "CA", "C", "O", "OXT"}

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.cif"
_RCSB_ASSEMBLY_URL = "https://files.rcsb.org/download/{pdb_id}-assembly1.cif"

#: A contiguous run at least this long, averaging below
#: :data:`DISORDER_PLDDT`, is a modelling failure rather than a surface: its
#: coordinates are an extended tether and the RSA computed from them is
#: meaningless.
DISORDER_MIN_LENGTH = 10
DISORDER_PLDDT = 50.0
_AFDB_URL = "https://alphafold.ebi.ac.uk/files/{accession}-model_v4.cif"


class StructureError(Exception):
    """Raised when a structure cannot be parsed or does not match the sequence."""


@dataclass(frozen=True, order=True)
class ResidueKey:
    """A residue in author numbering: chain, sequence number, insertion code."""

    chain: str
    resseq: int
    icode: str = " "

    @property
    def label(self) -> str:
        code = self.icode.strip()
        return f"{self.resseq}{code}" if code else str(self.resseq)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.chain}/{self.label}"


@dataclass
class ResidueRecord:
    """Per-residue structural facts, keyed by author numbering."""

    key: ResidueKey
    aa: str
    resname: str
    sasa: float = float("nan")
    sasa_context: float = float("nan")
    rsa: float = float("nan")
    plddt: float = float("nan")
    bfactor_mean: float = float("nan")
    centroid: Optional[Tuple[float, float, float]] = None
    centroid_source: str = "none"
    ca: Optional[Tuple[float, float, float]] = None
    secondary_structure: str = "-"
    has_altloc: bool = False
    in_disordered_region: bool = False
    #: RSA as computed, kept even where :attr:`rsa` is voided for disorder, so
    #: --keep-disordered has something to restore.
    rsa_raw: float = float("nan")

    @property
    def buried_by_context(self) -> bool:
        """True if extra chains occlude >20% of this residue's monomer SASA."""
        if self.sasa != self.sasa or self.sasa_context != self.sasa_context:
            return False
        if self.sasa <= 1e-6:
            return False
        return (self.sasa - self.sasa_context) / self.sasa > 0.20


@dataclass
class StructureModel:
    """A parsed structure chain plus everything computed from it."""

    path: Path
    chain_id: str
    residues: List[ResidueRecord] = field(default_factory=list)
    is_alphafold: bool = False
    context_chains: List[str] = field(default_factory=list)
    dssp_used: bool = False
    sasa_backend: str = "shrake_rupley"
    assembly: str = "asymmetric unit"
    disordered_regions: List[Tuple[ResidueKey, ResidueKey, float, int]] = field(
        default_factory=list
    )
    warnings: List[str] = field(default_factory=list)

    @property
    def sequence(self) -> str:
        return "".join(r.aa for r in self.residues)

    def by_key(self) -> Dict[ResidueKey, ResidueRecord]:
        return {r.key: r for r in self.residues}


# --------------------------------------------------------------------------
# fetching / parsing
# --------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover
        raise StructureError(f"cannot download {url}: requests is not installed") from exc
    response = requests.get(url, timeout=120)
    if response.status_code != 200:
        raise StructureError(f"download failed ({response.status_code}): {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


def resolve_structure(
    spec: str, cache_dir: Optional[Path] = None, prefer_assembly: bool = True
) -> Path:
    """Resolve a path, a 4-character PDB ID, or an AlphaFold DB accession.

    For a PDB ID the biological assembly is preferred over the asymmetric unit,
    since the asymmetric unit of an obligate oligomer leaves interface residues
    looking solvent-exposed.
    """
    path = Path(spec)
    if path.exists():
        return path
    cache_dir = Path(cache_dir or Path(tempfile.gettempdir()) / "epitope_map_cache")
    token = spec.strip()
    if token.upper().startswith("AF-") or token.upper().startswith("AF_"):
        accession = token.upper().replace("AF_", "AF-")
        if not accession.endswith(("-F1", "-F2", "-F3")):
            accession = f"{accession}-F1"
        dest = cache_dir / f"{accession}.cif"
        if not dest.exists():
            _download(_AFDB_URL.format(accession=accession), dest)
        return dest
    if len(token) == 4 and token[0].isdigit():
        # the biological assembly, not the asymmetric unit: an obligate dimer
        # analysed as a monomer reports interface residues as exposed
        if prefer_assembly:
            dest = cache_dir / f"{token.lower()}-assembly1.cif"
            if dest.exists():
                return dest
            try:
                return _download(
                    _RCSB_ASSEMBLY_URL.format(pdb_id=token.lower()), dest
                )
            except StructureError:
                pass
        dest = cache_dir / f"{token.lower()}.cif"
        if not dest.exists():
            _download(_RCSB_URL.format(pdb_id=token.lower()), dest)
        return dest
    raise StructureError(
        f"structure {spec!r} is not an existing file, a 4-character PDB ID, "
        "or an AlphaFold accession (AF-XXXXXX-F1)"
    )


def _open_structure(path: Path):
    text_path = path
    tmp: Optional[Path] = None
    if path.suffix == ".gz":
        tmp = Path(tempfile.mkdtemp()) / path.stem
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        text_path = tmp
    suffix = text_path.suffix.lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if suffix in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", str(text_path))
    return structure, text_path


def _is_alphafold(path: Path, structure) -> bool:
    name = path.name.upper()
    if name.startswith("AF-") or "ALPHAFOLD" in name:
        return True
    try:
        header = structure.header.get("name", "") or ""
    except Exception:  # pragma: no cover - defensive
        header = ""
    return "alphafold" in str(header).lower()


def _residue_atoms(residue, include_hydrogens: bool = False):
    for atom in residue:
        if not include_hydrogens and atom.element == "H":
            continue
        yield atom


def _centroid(residue) -> Tuple[Optional[Tuple[float, float, float]], str]:
    """Side-chain centroid, falling back to CB then CA."""
    coords = [
        atom.coord
        for atom in _residue_atoms(residue)
        if atom.get_id() not in _BACKBONE
    ]
    if coords:
        x = sum(float(c[0]) for c in coords) / len(coords)
        y = sum(float(c[1]) for c in coords) / len(coords)
        z = sum(float(c[2]) for c in coords) / len(coords)
        return (x, y, z), "sidechain"
    for atom_id, source in (("CB", "CB"), ("CA", "CA")):
        if atom_id in residue:
            c = residue[atom_id].coord
            return (float(c[0]), float(c[1]), float(c[2])), source
    return None, "none"


def _clean_altlocs(structure) -> bool:
    """Keep the highest-occupancy altloc per atom. Returns True if any existed."""
    found = False
    for atom in list(structure.get_atoms()):
        if atom.is_disordered():
            found = True
            try:
                atom.disordered_select(
                    max(atom.disordered_get_id_list(),
                        key=lambda alt: atom.disordered_get(alt).get_occupancy() or 0.0)
                )
            except Exception:  # pragma: no cover - defensive
                pass
    return found


def _protein_residues(chain) -> List:
    out = []
    for residue in chain:
        het, _, _ = residue.get_id()
        resname = residue.get_resname().strip().upper()
        if resname == "HOH":
            continue
        if het.strip() and resname not in _THREE_TO_ONE:
            continue
        if resname not in _THREE_TO_ONE and not is_aa(residue, standard=False):
            continue
        if "CA" not in residue:
            continue
        out.append(residue)
    return out


def _first_protein_chain(model) -> str:
    for chain in model:
        if len(_protein_residues(chain)) >= 20:
            return chain.get_id()
    for chain in model:
        if _protein_residues(chain):
            return chain.get_id()
    raise StructureError("no protein chain found in the structure")


# --------------------------------------------------------------------------
# SASA
# --------------------------------------------------------------------------


def _sasa_freesasa(structure_path: Path, chain_ids: Sequence[str]) -> Optional[Dict[ResidueKey, float]]:
    try:
        import freesasa
    except ImportError:
        return None
    try:  # pragma: no cover - exercised only where freesasa is installed
        freesasa.setVerbosity(freesasa.silent)
        struct = freesasa.Structure(str(structure_path))
        result = freesasa.calc(struct)
        residue_areas = result.residueAreas()
    except Exception:
        return None
    out: Dict[ResidueKey, float] = {}
    for chain, residues in residue_areas.items():
        if chain_ids and chain not in chain_ids:
            continue
        for number, area in residues.items():
            number = number.strip()
            icode = " "
            if number and not number[-1].isdigit():
                icode = number[-1]
                number = number[:-1]
            try:
                resseq = int(number)
            except ValueError:
                continue
            out[ResidueKey(chain, resseq, icode)] = float(area.total)
    return out or None


def _sasa_shrake_rupley(entity) -> Dict[ResidueKey, float]:
    ShrakeRupley().compute(entity, level="R")
    out: Dict[ResidueKey, float] = {}
    for chain in entity:
        for residue in chain:
            het, resseq, icode = residue.get_id()
            sasa = getattr(residue, "sasa", None)
            if sasa is None:
                continue
            out[ResidueKey(chain.get_id(), int(resseq), icode)] = float(sasa)
    return out


class _ChainSelector:
    """Copy of a model restricted to selected chains (for SASA in context)."""

    def __init__(self, model, chain_ids: Sequence[str]):
        self.model = model
        self.chain_ids = set(chain_ids)

    def build(self):
        import copy

        clone = copy.deepcopy(self.model)
        for chain in list(clone):
            if chain.get_id() not in self.chain_ids:
                clone.detach_child(chain.get_id())
            else:
                for residue in list(chain):
                    het, _, _ = residue.get_id()
                    resname = residue.get_resname().strip().upper()
                    if resname == "HOH" or (het.strip() and resname not in _THREE_TO_ONE):
                        chain.detach_child(residue.get_id())
        return clone


# --------------------------------------------------------------------------
# DSSP
# --------------------------------------------------------------------------


def _run_dssp(path: Path, model, chain_id: str) -> Dict[ResidueKey, str]:
    executable = None
    for candidate in ("mkdssp", "dssp"):
        if shutil.which(candidate):
            executable = candidate
            break
    if executable is None:
        return {}
    try:  # pragma: no cover - depends on an external binary
        from Bio.PDB.DSSP import DSSP

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dssp = DSSP(model, str(path), dssp=executable)
    except Exception:
        return {}
    out: Dict[ResidueKey, str] = {}
    for (chain, res_id), value in dssp.property_dict.items():
        if chain != chain_id:
            continue
        _, resseq, icode = res_id
        out[ResidueKey(chain, int(resseq), icode)] = value[2] or "-"
    return out


def _fallback_secondary_structure(residues: List[ResidueRecord]) -> None:
    """Mark nothing rather than guess; callers check ``dssp_used``."""
    for residue in residues:
        residue.secondary_structure = "-"


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------


def load_structure(
    spec: str,
    chain_id: Optional[str] = None,
    context_spec: Optional[str] = None,
    context_chains: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
    run_dssp: bool = True,
    prefer_assembly: bool = True,
) -> StructureModel:
    """Parse a structure and compute SASA, RSA, pLDDT, geometry and DSSP.

    ``context_spec``/``context_chains`` add extra chains (from the same file or a
    second structure) to a second SASA pass, so oligomer interfaces can be
    detected as residues that lose accessibility in context.
    """
    path = resolve_structure(spec, cache_dir=cache_dir, prefer_assembly=prefer_assembly)
    structure, text_path = _open_structure(path)
    model = next(iter(structure))
    had_altloc = _clean_altlocs(structure)

    if chain_id is None:
        chain_id = _first_protein_chain(model)
    if chain_id not in [c.get_id() for c in model]:
        raise StructureError(
            f"chain {chain_id!r} not in structure; chains present: "
            f"{[c.get_id() for c in model]}"
        )

    chain = model[chain_id]
    residues = _protein_residues(chain)
    if not residues:
        raise StructureError(f"chain {chain_id!r} contains no protein residues")

    is_af = _is_alphafold(path, structure)
    result = StructureModel(path=path, chain_id=chain_id, is_alphafold=is_af)
    if "assembly" in path.name:
        result.assembly = "biological assembly 1"
        if len([c for c in model]) > 1:
            result.warnings.append(
                f"using {result.assembly}: chains "
                f"{[c.get_id() for c in model]} are present, so pass "
                "--context-chains to have their interfaces masked"
            )
    if had_altloc:
        result.warnings.append(
            "structure contains altloc records; the highest-occupancy conformer "
            "was used for every atom"
        )

    # ---- monomer SASA
    monomer = _ChainSelector(model, [chain_id]).build()
    sasa_map = _sasa_freesasa(text_path, [chain_id])
    if sasa_map is not None:
        result.sasa_backend = "freesasa"
    else:
        sasa_map = _sasa_shrake_rupley(monomer)
        result.sasa_backend = "shrake_rupley"

    # ---- SASA in assembly context
    context_ids: List[str] = []
    context_model = None
    if context_chains:
        context_ids = [c for c in context_chains if c != chain_id]
    if context_spec:
        ctx_path = resolve_structure(context_spec, cache_dir=cache_dir)
        ctx_structure, _ = _open_structure(ctx_path)
        ctx_model = next(iter(ctx_structure))
        _clean_altlocs(ctx_structure)
        import copy

        context_model = copy.deepcopy(monomer)
        for ctx_chain in ctx_model:
            if context_ids and ctx_chain.get_id() not in context_ids:
                continue
            clone = copy.deepcopy(ctx_chain)
            # waters and ligands would inflate the apparent burial
            for residue in list(clone):
                het, _, _ = residue.get_id()
                resname = residue.get_resname().strip().upper()
                if resname == "HOH" or (het.strip() and resname not in _THREE_TO_ONE):
                    clone.detach_child(residue.get_id())
            if not list(clone):
                continue
            while clone.get_id() in [c.get_id() for c in context_model]:
                clone.id = f"{clone.get_id()}x"
            context_model.add(clone)
    elif context_ids:
        context_model = _ChainSelector(model, [chain_id, *context_ids]).build()

    sasa_context_map: Dict[ResidueKey, float] = {}
    if context_model is not None:
        sasa_context_map = _sasa_shrake_rupley(context_model)
        result.context_chains = context_ids or [
            c.get_id() for c in context_model if c.get_id() != chain_id
        ]

    # ---- per-residue records
    for residue in residues:
        het, resseq, icode = residue.get_id()
        resname = residue.get_resname().strip().upper()
        key = ResidueKey(chain_id, int(resseq), icode)
        aa = _THREE_TO_ONE.get(resname, "X")
        centroid, source = _centroid(residue)
        ca = None
        if "CA" in residue:
            c = residue["CA"].coord
            ca = (float(c[0]), float(c[1]), float(c[2]))
        bfactors = [float(a.get_bfactor()) for a in _residue_atoms(residue)]
        record = ResidueRecord(
            key=key,
            aa=aa,
            resname=resname,
            sasa=sasa_map.get(key, float("nan")),
            sasa_context=sasa_context_map.get(key, float("nan")),
            centroid=centroid,
            centroid_source=source,
            ca=ca,
            bfactor_mean=sum(bfactors) / len(bfactors) if bfactors else float("nan"),
            has_altloc=any(a.is_disordered() for a in residue),
        )
        record.rsa = record.rsa_raw = relative_sasa(aa, record.sasa)
        if is_af:
            record.plddt = record.bfactor_mean
        result.residues.append(record)

    if is_af:
        values = [r.plddt for r in result.residues if r.plddt == r.plddt]
        if values and (max(values) > 100.5 or min(values) < 0):
            result.warnings.append(
                "B-factors are outside 0-100, so they were not treated as pLDDT "
                "despite the file looking like an AlphaFold model"
            )
            for r in result.residues:
                r.plddt = float("nan")
            result.is_alphafold = False

    if run_dssp:
        ss_map = _run_dssp(text_path, model, chain_id)
        if ss_map:
            result.dssp_used = True
            for record in result.residues:
                record.secondary_structure = ss_map.get(record.key, "-")
        else:
            _fallback_secondary_structure(result.residues)
            result.warnings.append(
                "DSSP (mkdssp) not available: secondary structure is unassigned, so "
                "chimera boundaries are proposed from geometry alone and may cut "
                "through a helix or strand"
            )
    else:
        _fallback_secondary_structure(result.residues)

    _flag_disordered_regions(result)

    missing = _detect_internal_gaps(result.residues)
    if missing:
        result.warnings.append(
            f"{len(missing)} internal residue-numbering gap(s) in chain {chain_id} "
            f"(unmodelled regions): {', '.join(missing[:8])}"
            + (" ..." if len(missing) > 8 else "")
        )
    return result


def _flag_disordered_regions(
    model: StructureModel,
    min_length: int = DISORDER_MIN_LENGTH,
    cutoff: float = DISORDER_PLDDT,
) -> None:
    """Mark contiguous low-pLDDT runs, and void the RSA computed inside them.

    A single low-pLDDT residue in an otherwise ordered loop is worth keeping -
    flexible loops are where epitopes sit. A thirty-residue stretch averaging
    pLDDT 35 is different in kind: AlphaFold has modelled it as an extended
    tether, every residue in it looks fully exposed, and the whole region floods
    the candidate list. Those residues keep their sequence scores but lose their
    (meaningless) RSA.
    """
    if not model.is_alphafold:
        return
    residues = model.residues
    low = [r.plddt == r.plddt and r.plddt < cutoff for r in residues]

    start = None
    for index in range(len(residues) + 1):
        inside = index < len(residues) and low[index]
        if inside and start is None:
            start = index
        elif not inside and start is not None:
            run = residues[start:index]
            values = [r.plddt for r in run if r.plddt == r.plddt]
            if len(run) >= min_length and values and sum(values) / len(values) < cutoff:
                for residue in run:
                    residue.in_disordered_region = True
                    residue.rsa = float("nan")  # not a surface, so not an RSA
                model.disordered_regions.append(
                    (run[0].key, run[-1].key, sum(values) / len(values), len(run))
                )
            start = None

    if model.disordered_regions:
        described = ", ".join(
            f"{a.label}-{b.label} ({n} residues, mean pLDDT {mean:.0f})"
            for a, b, mean, n in model.disordered_regions
        )
        model.warnings.append(
            f"{len(model.disordered_regions)} disordered region(s) in the model: "
            f"{described}. AlphaFold models these as extended tethers, so their "
            "RSA is meaningless and has been voided; they are excluded from patch "
            "seeding unless --keep-disordered is given"
        )


def _detect_internal_gaps(residues: List[ResidueRecord]) -> List[str]:
    gaps: List[str] = []
    for prev, curr in zip(residues, residues[1:]):
        if curr.key.resseq > prev.key.resseq + 1:
            gaps.append(f"{prev.key.label}->{curr.key.label}")
    return gaps


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
