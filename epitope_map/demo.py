"""A deterministic worked example: a planted epitope on a toy structure.

This is what ``epitope-map --demo`` runs, what ``examples/`` writes out, and what
the test suite checks against. Nothing here is fetched or random: the structure
is a packed ball of residues with a buried core, a chosen set of sequence
positions is placed inside one surface cap, and the non-binder species carry
drastic substitutions there plus neutral drift elsewhere. A correct pipeline
should rank the cap first, which is exactly what a real run should do with a
real epitope.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

AA = "ACDEFGHIKLMNPQRSTVWY"
THREE = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY",
    "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
    "W": "TRP", "Y": "TYR",
}

N_RESIDUES = 120
FIRST_RESSEQ = 25  # deliberate offset from 1-based sequence numbering
LATTICE_SPACING = 5.6
NOISE_KEEPOUT = 16.0  # A between the planted cap and any neutral clade marker
CORE_INDICES = list(range(8, 28))  # buried residues
EPITOPE_INDICES = [40, 41, 43, 45, 47, 90, 92, 94]  # planted, spatially clustered
NOISE_INDICES = [3, 5, 30, 33, 35, 55, 58, 62, 66, 70, 74, 78, 82, 100, 104, 108, 112, 116]

# residue 100 in author numbering is followed by 100A, to exercise insertion codes
INSERTION_AFTER_INDEX = 75
# residues absent from the ATOM records, to exercise SEQRES-vs-ATOM gaps
UNMODELLED_INDICES = [60, 61, 62]

#: Optional "stalk": a long, highly divergent, low-pLDDT stretch. AlphaFold
#: models such regions as extended tethers, which inflates their RSA so every
#: residue passes the exposure filter, and they are the least constrained part
#: of a protein so they look discriminating too. Together that floods the
#: candidate list with a modelling artifact - the failure the disordered-region
#: filter exists to catch.
STALK_INDICES = list(range(55, 85))
STALK_PLDDT = 30.0
ORDERED_PLDDT = 88.0


def _fcc_ball(n: int, spacing: float) -> List[Tuple[float, float, float]]:
    """The ``n`` innermost points of a face-centred-cubic lattice.

    A packed ball, not a hollow sphere: interior residues are then genuinely
    buried (RSA well under the 0.20 cutoff) while the outer layer is exposed,
    which is what makes the burial filter testable.
    """
    import itertools

    step = spacing / math.sqrt(2.0)
    reach = int(math.ceil((n ** (1.0 / 3.0)) * 1.5)) + 2
    points = []
    for i, j, k in itertools.product(range(-reach, reach + 1), repeat=3):
        if (i + j + k) % 2:
            continue
        points.append((i * step, j * step, k * step))
    points.sort(key=lambda p: (sum(c * c for c in p), p))
    if len(points) < n:  # pragma: no cover - reach is generous
        raise RuntimeError("lattice too small")
    return points[:n]


def coordinates() -> Dict[int, Tuple[float, float, float]]:
    """Sequence index -> CA coordinate.

    The planted epitope occupies one surface cap; the neutral clade-marker
    positions are put on the far side of the ball, so the synthetic case is a
    clean positive control rather than a coin flip.
    """
    points = _fcc_ball(N_RESIDUES, LATTICE_SPACING)
    radius = max(math.sqrt(sum(c * c for c in p)) for p in points)
    core_points = points[: len(CORE_INDICES)]
    surface_points = list(points[len(CORE_INDICES) :])
    # order by "how far up the +y cap" a point sits, but only outer-shell points
    # qualify as cap seats, so every planted epitope residue is really exposed
    outer_cut = sorted(
        (math.sqrt(sum(c * c for c in p)) for p in surface_points), reverse=True
    )[min(len(surface_points) - 1, 45)]
    surface_points.sort(
        key=lambda p: (
            math.sqrt(sum(c * c for c in p)) < outer_cut,
            -(p[1] / (math.sqrt(sum(c * c for c in p)) or 1.0)),
        )
    )

    coords: Dict[int, Tuple[float, float, float]] = {}
    for index, point in zip(CORE_INDICES, core_points):
        coords[index] = point
    cap = surface_points[: len(EPITOPE_INDICES)]
    cap_centroid = tuple(sum(c[i] for c in cap) / len(cap) for i in range(3))
    rest = surface_points[len(EPITOPE_INDICES) :]
    # neutral clade markers are scattered evenly over the rest of the surface
    # (real drift is not clustered) and kept clear of the planted cap
    away = [
        point
        for point in rest
        if math.dist(point, cap_centroid) > NOISE_KEEPOUT
    ]
    stride = max(1, len(away) // len(NOISE_INDICES))
    far = away[::stride][: len(NOISE_INDICES)]
    if len(far) < len(NOISE_INDICES):  # pragma: no cover - defensive
        far = away[: len(NOISE_INDICES)]
    chosen = set(far)
    middle = [point for point in rest if point not in chosen]
    for index, point in zip(EPITOPE_INDICES, cap):
        coords[index] = point
    for index, point in zip(NOISE_INDICES, far):
        coords[index] = point
    remaining = [
        i
        for i in range(N_RESIDUES)
        if i not in CORE_INDICES and i not in EPITOPE_INDICES and i not in NOISE_INDICES
    ]
    for index, point in zip(remaining, middle):
        coords[index] = point
    assert len(coords) == N_RESIDUES, (len(coords), radius)
    return coords


def author_numbers() -> List[Tuple[int, str]]:
    """Sequence index -> (resseq, icode) in author numbering."""
    numbers: List[Tuple[int, str]] = []
    resseq = FIRST_RESSEQ
    for index in range(N_RESIDUES):
        if index == INSERTION_AFTER_INDEX:
            numbers.append((resseq - 1, "A"))  # shares the previous number
        else:
            numbers.append((resseq, " "))
            resseq += 1
    return numbers


def reference_sequence(seed: int = 7) -> str:
    rng = random.Random(seed)
    seq = [rng.choice("AVLIFMSTNQKRDEGY") for _ in range(N_RESIDUES)]
    for index in CORE_INDICES:
        seq[index] = rng.choice("AVLIFM")  # hydrophobic core
    for index in EPITOPE_INDICES:
        seq[index] = rng.choice("KRDEQNST")  # polar surface
    seq[45] = "K"
    seq[47] = "D"
    return "".join(seq)


def species_sequences(
    include_informative: bool = False,
    disordered_stalk: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return ``(sequences, binding_calls)`` for the synthetic panel.

    With ``include_informative`` an extra rodent is added that carries the
    planted epitope substitutions on an otherwise rodent background, so the
    binding pattern no longer follows the sequence tree. That is the "one more
    informative species" case the report keeps recommending, and
    ``validation/benchmark.py scaling`` measures what it buys.
    """
    rng = random.Random(11)
    mouse = reference_sequence()
    seqs = {"mouse": mouse}

    rat = list(mouse)
    for index in NOISE_INDICES[:6]:
        rat[index] = _swap(rat[index], rng, conservative=True)
    seqs["rat"] = "".join(rat)

    hamster = list(mouse)
    for index in NOISE_INDICES[3:10]:
        hamster[index] = _swap(hamster[index], rng, conservative=True)
    seqs["hamster"] = "".join(hamster)

    drastic = {40: "W", 41: "P", 43: "W", 45: "E", 47: "R", 90: "G", 92: "W", 94: "P"}
    human = list(mouse)
    for index in NOISE_INDICES:
        human[index] = _swap(human[index], rng, conservative=True)
    for index, aa in drastic.items():
        human[index] = aa
    # a differential N-glycosylation sequon in the primates only
    human[64], human[65], human[66] = "N", "G", "T"
    seqs["human"] = "".join(human)

    marmoset = list(human)
    for index in NOISE_INDICES[::3]:
        marmoset[index] = _swap(marmoset[index], rng, conservative=True)
    seqs["marmoset"] = "".join(marmoset)

    macaque = list(human)
    for index in NOISE_INDICES[1::3]:
        macaque[index] = _swap(macaque[index], rng, conservative=True)
    # an indel relative to the rodents
    macaque = macaque[:110] + macaque[113:]
    seqs["macaque"] = "".join(macaque)

    if disordered_stalk:
        # the stalk diverges hard between the two clades, as a real one does
        stalk_rng = random.Random(23)
        for name in ("human", "marmoset", "macaque"):
            sequence = list(seqs[name])
            for index in STALK_INDICES:
                if index < len(sequence):
                    sequence[index] = _swap(sequence[index], stalk_rng)
            seqs[name] = "".join(sequence)

    if include_informative:
        vole = list(mouse)
        for index, aa in drastic.items():
            vole[index] = aa
        seqs["vole"] = "".join(vole)

    calls = {
        "mouse": "binder",
        "rat": "binder",
        "hamster": "binder",
        "human": "non_binder",
        "marmoset": "non_binder",
        "macaque": "non_binder",
    }
    if include_informative:
        calls["vole"] = "non_binder"
    return seqs, calls


def _swap(aa: str, rng: random.Random, conservative: bool = False) -> str:
    groups = ["AVLIM", "FYW", "STNQ", "KR", "DE", "G", "P", "C", "H"]
    if conservative:
        for group in groups:
            if aa in group and len(group) > 1:
                return rng.choice(group.replace(aa, ""))
        return aa
    return rng.choice(AA.replace(aa, ""))


def write_pdb(
    path: Path, sequence: str | None = None, disordered_stalk: bool = False
) -> Path:
    """Write the toy structure for the reference sequence.

    With ``disordered_stalk`` the B-factor column carries pLDDT values - high
    everywhere except the stalk - and the file is named so it is recognised as
    an AlphaFold model.
    """
    sequence = sequence or reference_sequence()
    coords = coordinates()
    numbers = author_numbers()
    lines: List[str] = ["HEADER    SYNTHETIC TEST STRUCTURE"]
    serial = 1
    for index, aa in enumerate(sequence):
        if index in UNMODELLED_INDICES:
            continue
        resseq, icode = numbers[index]
        x, y, z = coords[index]
        # a small spray of atoms around the CA so SASA has something to chew on
        atoms = [
            ("N", (x - 1.2, y, z)),
            ("CA", (x, y, z)),
            ("C", (x + 1.2, y, z)),
            ("O", (x + 1.6, y + 1.0, z)),
        ]
        if aa != "G":
            direction = _unit((x, y, z))
            # surface residues carry a long radial side chain (which is what
            # makes the outer layer solvent-exposed and the interior packed);
            # core residues keep a short one so they stay buried
            reaches = (
                ((1.4,) if index in CORE_INDICES else (1.6, 3.0, 4.2))
            )
            for name, reach in zip(("CB", "CG", "CD"), reaches):
                atoms.append(
                    (
                        name,
                        (
                            x + direction[0] * reach,
                            y + direction[1] * reach,
                            z + direction[2] * reach,
                        ),
                    )
                )
        bfactor = (
            STALK_PLDDT
            if disordered_stalk and index in STALK_INDICES
            else ORDERED_PLDDT
        )
        for name, (ax, ay, az) in atoms:
            element = name[0]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s}{THREE[aa]:>4s} A{resseq:4d}{icode}"
                f"   {ax:8.3f}{ay:8.3f}{az:8.3f}  1.00{bfactor:6.2f}          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _unit(v: Sequence[float]) -> Tuple[float, float, float]:
    length = math.sqrt(sum(c * c for c in v)) or 1.0
    return (v[0] / length, v[1] / length, v[2] / length)


def write_inputs(
    directory: Path,
    include_informative: bool = False,
    disordered_stalk: bool = False,
) -> Dict[str, Path]:
    """Write fasta, binding CSV and PDB into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seqs, calls = species_sequences(
        include_informative=include_informative, disordered_stalk=disordered_stalk
    )
    fasta = directory / "species.fasta"
    with open(fasta, "w") as handle:
        for name, seq in seqs.items():
            handle.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                handle.write(seq[i : i + 60] + "\n")
    binding = directory / "binding.csv"
    binding.write_text(
        "species,binding\n" + "".join(f"{k},{v}\n" for k, v in calls.items())
    )
    name = "AF-DEMO-F1.pdb" if disordered_stalk else "reference.pdb"
    pdb = write_pdb(
        directory / name, sequence=seqs["mouse"], disordered_stalk=disordered_stalk
    )
    return {"sequences": fasta, "binding": binding, "structure": pdb}


def truth_numbers() -> List[str]:
    """Author-numbering labels of the planted epitope residues."""
    numbers = author_numbers()
    out = []
    for index in EPITOPE_INDICES:
        resseq, icode = numbers[index]
        code = icode.strip()
        out.append(f"{resseq}{code}" if code else str(resseq))
    return out


def write_species_structure(
    path: Path, species: str = "human", shift: int = 0, first: int = 0
) -> Path:
    """A toy structure for another species, on the same fold as the reference.

    Same coordinates (the fold is conserved, as it is between real orthologs),
    that species' own sequence, and its own numbering - which is what makes it a
    fair test of deriving equivalences from geometry instead of from the
    alignment. ``shift`` renumbers the file and ``first`` truncates the modelled
    region, the two ways real structures differ from their sequence entries.
    """
    seqs, _ = species_sequences()
    sequence = seqs[species]
    coords = coordinates()
    lines: List[str] = ["HEADER    SYNTHETIC ORTHOLOG STRUCTURE"]
    serial = 1
    for index, aa in enumerate(sequence):
        if index not in coords or index < first:
            continue
        x, y, z = coords[index]
        atoms = [
            ("N", (x - 1.2, y, z)),
            ("CA", (x, y, z)),
            ("C", (x + 1.2, y, z)),
            ("O", (x + 1.6, y + 1.0, z)),
        ]
        if aa != "G":
            direction = _unit((x, y, z))
            atoms.append(
                ("CB", (x + direction[0] * 1.6, y + direction[1] * 1.6,
                        z + direction[2] * 1.6))
            )
        for name, (ax, ay, az) in atoms:
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s}{THREE[aa]:>4s} B{index + 1 + shift:4d} "
                f"   {ax:8.3f}{ay:8.3f}{az:8.3f}  1.00 80.00          {name[0]:>2s}"
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path
