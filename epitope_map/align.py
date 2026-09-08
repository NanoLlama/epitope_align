"""Multiple sequence alignment and the one object that owns all numbering.

Numbering is the main source of bugs in this kind of tool, so exactly one class
converts between the three coordinate systems in play:

* **alignment column** - index into the MSA
* **reference index**  - 0-based position in the ungapped reference sequence
* **reference number** - author numbering of the reference structure (may carry
  insertion codes, may start at an arbitrary offset, may skip unmodelled runs)

Nothing outside :class:`ResidueMap` is allowed to do index arithmetic.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .io_seq import SpeciesRecord
from .structure import ResidueKey, StructureModel

GAP = "-"


class AlignmentError(Exception):
    """Raised when alignment or the sequence/structure map cannot be built."""


@dataclass
class Alignment:
    """An MSA plus provenance and per-species identity to the reference."""

    sequences: Dict[str, str]
    reference: str
    method: str
    identities: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(next(iter(self.sequences.values()))) if self.sequences else 0

    def column(self, index: int) -> Dict[str, str]:
        return {name: seq[index] for name, seq in self.sequences.items()}

    def to_fasta(self) -> str:
        lines = []
        for name, seq in self.sequences.items():
            lines.append(f">{name}")
            for i in range(0, len(seq), 60):
                lines.append(seq[i : i + 60])
        return "\n".join(lines) + "\n"


def _write_fasta(records: Sequence[SpeciesRecord], path: Path) -> None:
    with open(path, "w") as handle:
        for record in records:
            handle.write(f">{record.name}\n")
            seq = record.sequence
            for i in range(0, len(seq), 60):
                handle.write(seq[i : i + 60] + "\n")


def _parse_aligned_fasta(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    name: Optional[str] = None
    chunks: List[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(chunks).upper()
            name = line[1:].strip().split()[0]
            chunks = []
        elif name is not None:
            chunks.append(line.strip())
    if name is not None:
        out[name] = "".join(chunks).upper()
    return out


def _run_mafft(
    records: Sequence[SpeciesRecord],
    threads: int = 1,
    extra_args: Optional[Sequence[str]] = None,
) -> Optional[Dict[str, str]]:
    if not shutil.which("mafft"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        infile = Path(tmp) / "in.fasta"
        _write_fasta(records, infile)
        try:
            proc = subprocess.run(
                ["mafft", *(extra_args or ["--auto"]), "--anysymbol",
                 "--thread", str(threads), str(infile)],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        return _parse_aligned_fasta(proc.stdout)


def _run_muscle(records: Sequence[SpeciesRecord]) -> Optional[Dict[str, str]]:
    if not shutil.which("muscle"):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        infile = Path(tmp) / "in.fasta"
        outfile = Path(tmp) / "out.fasta"
        _write_fasta(records, infile)
        for argv in (
            ["muscle", "-align", str(infile), "-output", str(outfile)],  # v5
            ["muscle", "-in", str(infile), "-out", str(outfile)],  # v3
        ):
            try:
                proc = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
            except (OSError, subprocess.TimeoutExpired):
                continue
            if proc.returncode == 0 and outfile.exists() and outfile.stat().st_size:
                return _parse_aligned_fasta(outfile.read_text())
        return None


def _free_end_gaps(aligner) -> None:
    """Make terminal gaps free, across Biopython naming changes.

    Biopython 1.85+ renamed ``target_end_gap_score``/``query_end_gap_score`` to
    ``end_insertion_score``/``end_deletion_score``; setting the old names still
    works but emits a deprecation warning.
    """
    for new, old in (
        ("end_insertion_score", "target_end_gap_score"),
        ("end_deletion_score", "query_end_gap_score"),
    ):
        try:
            setattr(aligner, new, 0.0)
        except (AttributeError, ValueError):  # pragma: no cover - older Biopython
            setattr(aligner, old, 0.0)


def _pairwise_to_reference(
    records: Sequence[SpeciesRecord],
    reference: str,
    open_gap: float = -11,
    extend_gap: float = -1,
    matrix: str = "BLOSUM62",
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Fallback: pairwise-align everything to the reference, keep ref columns.

    Insertions relative to the reference are dropped, so the resulting 'MSA' has
    exactly one column per reference residue. This is a real degradation; the
    number of residues discarded per species is returned so the caller can say
    how much was lost rather than only that the method was crude.
    """
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = open_gap
    aligner.extend_gap_score = extend_gap
    _free_end_gaps(aligner)
    try:
        from Bio.Align import substitution_matrices

        aligner.substitution_matrix = substitution_matrices.load(matrix)
    except Exception:  # pragma: no cover - defensive
        aligner.match_score = 2
        aligner.mismatch_score = -1

    ref_seq = next(r.sequence for r in records if r.name == reference)
    aligned = {reference: ref_seq}
    dropped: Dict[str, int] = {}
    for record in records:
        if record.name == reference:
            continue
        alignment = aligner.align(ref_seq, record.sequence)[0]
        projected = [GAP] * len(ref_seq)
        kept = 0
        for (ref_start, ref_end), (qry_start, qry_end) in zip(*alignment.aligned):
            for offset in range(ref_end - ref_start):
                projected[ref_start + offset] = record.sequence[qry_start + offset]
                kept += 1
        aligned[record.name] = "".join(projected)
        dropped[record.name] = len(record.sequence) - kept
    return aligned, dropped


def _is_subsequence(small: str, big: str) -> bool:
    """Is every residue of ``small`` present in ``big``, in order?"""
    iterator = iter(big)
    return all(character in iterator for character in small)


def percent_identity(a: str, b: str) -> float:
    """Identity over columns where both sequences have a residue."""
    both = [(x, y) for x, y in zip(a, b) if x != GAP and y != GAP]
    if not both:
        return 0.0
    return 100.0 * sum(1 for x, y in both if x == y) / len(both)


def align_sequences(
    records: Sequence[SpeciesRecord],
    reference: str,
    method: str = "auto",
    threads: int = 1,
    low_identity_threshold: float = 40.0,
) -> Alignment:
    """Align with MAFFT, else MUSCLE, else pairwise-to-reference projection."""
    aligned: Optional[Dict[str, str]] = None
    used = ""
    warnings_: List[str] = []

    if method in ("auto", "mafft"):
        aligned = _run_mafft(records, threads=threads)
        used = "mafft --auto" if aligned else ""
    if aligned is None and method in ("auto", "muscle"):
        aligned = _run_muscle(records)
        used = "muscle" if aligned else ""
    dropped: Dict[str, int] = {}
    if aligned is None:
        if method not in ("auto", "pairwise"):
            raise AlignmentError(f"requested aligner {method!r} is not available on PATH")
        aligned, dropped = _pairwise_to_reference(records, reference)
        used = "biopython-pairwise-to-reference"
        lost = {name: count for name, count in dropped.items() if count}
        warnings_.append(
            "NO TRUE MSA: neither MAFFT nor MUSCLE was found on PATH, so each "
            "sequence was pairwise-aligned to the reference and insertions "
            "relative to the reference were discarded"
            + (
                f" ({', '.join(f'{n}: {c} residue(s)' for n, c in lost.items())})"
                if lost
                else ""
            )
            + ". Install MAFFT before trusting any indel-driven result."
        )

    missing = {r.name for r in records} - set(aligned)
    if missing:
        raise AlignmentError(f"aligner did not return sequences for: {sorted(missing)}")
    aligned = {r.name: aligned[r.name] for r in records}

    lengths = {len(s) for s in aligned.values()}
    if len(lengths) != 1:
        raise AlignmentError(f"aligned sequences have unequal lengths: {sorted(lengths)}")

    # the aligner may upper/lower-case or reorder; verify nothing was mangled.
    # the pairwise fallback drops insertions by construction, so there the
    # requirement is that what remains is still in order and unaltered.
    for record in records:
        ungapped = aligned[record.name].replace(GAP, "").replace(".", "")
        original = record.sequence.upper()
        if ungapped == original:
            continue
        if dropped and _is_subsequence(ungapped, original):
            continue
        raise AlignmentError(
            f"aligned sequence for {record.name!r} does not match the input "
            "sequence once gaps are removed"
        )

    ref_aligned = aligned[reference]
    identities = {
        name: percent_identity(ref_aligned, seq)
        for name, seq in aligned.items()
        if name != reference
    }
    for name, identity in identities.items():
        if identity < low_identity_threshold:
            warnings_.append(
                f"{name} is only {identity:.1f}% identical to the reference "
                f"{reference} - check that it is the true ortholog and not a "
                "paralog or a misannotated accession"
            )
    return Alignment(
        sequences=aligned,
        reference=reference,
        method=used,
        identities=identities,
        warnings=warnings_,
    )


# --------------------------------------------------------------------------
# ResidueMap
# --------------------------------------------------------------------------


@dataclass
class Position:
    """One reference residue in every coordinate system at once."""

    ref_index: int
    column: int
    aa: str
    key: Optional[ResidueKey]
    number: Optional[str]
    in_ectodomain: bool
    modelled: bool


class ResidueMap:
    """Owns every conversion between column, reference index and author number."""

    def __init__(
        self,
        alignment: Alignment,
        ref_index_to_key: Dict[int, ResidueKey],
        ectodomain: Optional[Tuple[int, int]] = None,
        ectodomain_numbering: str = "structure",
        insertion_columns: Optional[Dict[int, List[int]]] = None,
        warnings: Optional[List[str]] = None,
    ):
        self.alignment = alignment
        self.reference = alignment.reference
        self.ref_seq = alignment.sequences[alignment.reference].replace(GAP, "")
        self._ref_index_to_key = dict(ref_index_to_key)
        self._key_to_ref_index = {v: k for k, v in ref_index_to_key.items()}
        self.ectodomain = ectodomain
        self.ectodomain_numbering = ectodomain_numbering
        self.insertion_columns = insertion_columns or {}
        self.warnings = list(warnings or [])

        ref_aligned = alignment.sequences[alignment.reference]
        self._col_of: List[int] = []
        self._refidx_of_col: Dict[int, int] = {}
        idx = 0
        for column, char in enumerate(ref_aligned):
            if char == GAP:
                continue
            self._col_of.append(column)
            self._refidx_of_col[column] = idx
            idx += 1
        if idx != len(self.ref_seq):  # pragma: no cover - defensive
            raise AlignmentError("reference column map is inconsistent")

        self._number_to_ref_index = {
            key.label: ref_index for ref_index, key in self._ref_index_to_key.items()
        }

    # ---- basic sizes -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.ref_seq)

    @property
    def n_columns(self) -> int:
        return self.alignment.length

    # ---- conversions -----------------------------------------------------
    def column_of(self, ref_index: int) -> int:
        return self._col_of[ref_index]

    def ref_index_of_column(self, column: int) -> Optional[int]:
        return self._refidx_of_col.get(column)

    def key_of(self, ref_index: int) -> Optional[ResidueKey]:
        return self._ref_index_to_key.get(ref_index)

    def ref_index_of_key(self, key: ResidueKey) -> Optional[int]:
        return self._key_to_ref_index.get(key)

    def number_of(self, ref_index: int) -> Optional[str]:
        key = self.key_of(ref_index)
        return key.label if key else None

    def ref_index_of_number(self, number: str) -> Optional[int]:
        return self._number_to_ref_index.get(str(number).strip())

    def aa_at(self, ref_index: int) -> str:
        return self.ref_seq[ref_index]

    def residue_at(self, species: str, ref_index: int) -> str:
        """Residue of ``species`` in the column of this reference position."""
        return self.alignment.sequences[species][self.column_of(ref_index)]

    def species_index(self, species: str, ref_index: int) -> Optional[int]:
        """1-based position in *that species'* own ungapped sequence, or None.

        Used for reciprocal mutants, which are made in the non-binder background
        and therefore must be numbered in the non-binder's own sequence.
        """
        aligned = self.alignment.sequences[species]
        column = self.column_of(ref_index)
        if aligned[column] == GAP:
            return None
        return sum(1 for c in aligned[: column + 1] if c != GAP)

    def is_modelled(self, ref_index: int) -> bool:
        return ref_index in self._ref_index_to_key

    # ---- ectodomain ------------------------------------------------------
    def in_ectodomain(self, ref_index: int) -> bool:
        if self.ectodomain is None:
            return True
        start, end = self.ectodomain
        if self.ectodomain_numbering == "sequence":
            return start <= ref_index + 1 <= end
        key = self.key_of(ref_index)
        if key is None:
            return False
        return start <= key.resseq <= end

    # ---- iteration -------------------------------------------------------
    def positions(self) -> Iterator[Position]:
        for ref_index in range(len(self.ref_seq)):
            key = self.key_of(ref_index)
            yield Position(
                ref_index=ref_index,
                column=self.column_of(ref_index),
                aa=self.aa_at(ref_index),
                key=key,
                number=key.label if key else None,
                in_ectodomain=self.in_ectodomain(ref_index),
                modelled=key is not None,
            )

    # ---- construction ----------------------------------------------------
    @classmethod
    def build(
        cls,
        alignment: Alignment,
        structure: StructureModel,
        ectodomain: Optional[Tuple[int, int]] = None,
        ectodomain_numbering: str = "structure",
        mismatch_tolerance: float = 0.05,
    ) -> "ResidueMap":
        """Anchor the reference sequence onto the structure's author numbering.

        Raises with a residue-level diff if the two disagree beyond
        ``mismatch_tolerance`` - proceeding on a bad map silently produces
        confident nonsense.
        """
        ref_seq = alignment.sequences[alignment.reference].replace(GAP, "")
        struct_seq = structure.sequence
        if not struct_seq:
            raise AlignmentError("structure chain has no residues to map onto")

        mapping, mismatches, aligned_pairs = _map_sequence_to_structure(
            ref_seq, struct_seq, structure
        )
        warnings_: List[str] = []
        if aligned_pairs == 0:
            raise AlignmentError(
                "could not align the reference sequence to the structure sequence"
            )
        mismatch_fraction = len(mismatches) / aligned_pairs
        if mismatch_fraction > mismatch_tolerance:
            preview = "\n".join(
                f"  ref {i + 1}{seq_aa}  vs  structure {key.label}{struct_aa}"
                for i, seq_aa, key, struct_aa in mismatches[:15]
            )
            raise AlignmentError(
                "reference sequence and structure disagree at "
                f"{len(mismatches)}/{aligned_pairs} aligned residues "
                f"({mismatch_fraction:.1%} > tolerance {mismatch_tolerance:.1%}).\n"
                "Is the structure really the reference species, and is the right "
                "chain selected?\nFirst mismatches:\n" + preview
            )
        if mismatches:
            warnings_.append(
                f"{len(mismatches)} residue(s) differ between the reference sequence "
                f"and the structure ({mismatch_fraction:.1%}); within tolerance, "
                "structural values for those positions are still used"
            )

        unmapped = len(ref_seq) - len(mapping)
        if unmapped:
            warnings_.append(
                f"{unmapped} reference residue(s) have no structural counterpart "
                "(unmodelled or outside the construct); they are kept in "
                "residues.tsv without structural values and cannot seed patches"
            )

        insertion_columns = _insertion_columns(alignment)
        offsets = {
            key.resseq - (ref_index + 1) for ref_index, key in mapping.items()
        }
        if len(offsets) == 1:
            offset = offsets.pop()
            if offset:
                warnings_.append(
                    f"structure author numbering runs {offset:+d} relative to the "
                    "reference sequence (e.g. a signal-peptide offset); "
                    f"--ectodomain is interpreted in {ectodomain_numbering} numbering"
                )
        return cls(
            alignment,
            mapping,
            ectodomain=ectodomain,
            ectodomain_numbering=ectodomain_numbering,
            insertion_columns=insertion_columns,
            warnings=warnings_,
        )


def _map_sequence_to_structure(
    ref_seq: str, struct_seq: str, structure: StructureModel
) -> Tuple[Dict[int, ResidueKey], List[Tuple[int, str, ResidueKey, str]], int]:
    """Global-align the reference sequence to the ATOM-record sequence."""
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -12
    aligner.extend_gap_score = -1
    _free_end_gaps(aligner)

    alignment = aligner.align(ref_seq, struct_seq)[0]
    keys = [r.key for r in structure.residues]
    mapping: Dict[int, ResidueKey] = {}
    mismatches: List[Tuple[int, str, ResidueKey, str]] = []
    aligned_pairs = 0
    for (ref_start, ref_end), (str_start, str_end) in zip(*alignment.aligned):
        for offset in range(ref_end - ref_start):
            ref_index = ref_start + offset
            struct_index = str_start + offset
            mapping[ref_index] = keys[struct_index]
            aligned_pairs += 1
            if ref_seq[ref_index] != struct_seq[struct_index]:
                mismatches.append(
                    (ref_index, ref_seq[ref_index], keys[struct_index], struct_seq[struct_index])
                )
    return mapping, mismatches, aligned_pairs


def _insertion_columns(alignment: Alignment) -> Dict[int, List[int]]:
    """Columns where the reference is gapped, bucketed by preceding ref index.

    These are insertions in one or more of the other species. They cannot be
    placed on the reference structure, but they are reported next to the
    reference residue they follow.
    """
    ref_aligned = alignment.sequences[alignment.reference]
    out: Dict[int, List[int]] = {}
    ref_index = -1
    for column, char in enumerate(ref_aligned):
        if char != GAP:
            ref_index += 1
        else:
            out.setdefault(ref_index, []).append(column)
    return out


# --------------------------------------------------------------------------
# alignment reliability
# --------------------------------------------------------------------------

#: Columns below this agreement rate are not trustworthy enough to name specific
#: residue equivalences from, which is what a point mutant is.
CONFIDENCE_CUTOFF = 0.7

#: Half-width of the window used for local identity.
IDENTITY_WINDOW = 20

#: A window this far (in percentage points) below the chain average is a region
#: where the aligner is guessing.
LOW_IDENTITY_MARGIN = 15.0


def _alternative_alignments(
    records: Sequence[SpeciesRecord], reference: str, threads: int = 1
) -> List[Dict[str, str]]:
    """Re-align the same sequences a few different ways.

    Where MAFFT and MUSCLE are installed, their genuinely different algorithms
    are used (local and global pair refinement). Where they are not, the
    pairwise fallback is re-run with different gap penalties and substitution
    matrices, which is weaker but still exposes the columns whose equivalences
    depend on the scoring choice rather than on the sequences.
    """
    variants: List[Dict[str, str]] = []

    for extra in (["--localpair", "--maxiterate", "100"], ["--globalpair", "--maxiterate", "100"]):
        aligned = _run_mafft(records, threads=threads, extra_args=extra)
        if aligned:
            variants.append(aligned)
    muscle = _run_muscle(records)
    if muscle:
        variants.append(muscle)

    if len(variants) < 2:
        # a deliberately wide spread: where the sequences are unambiguous every
        # setting recovers the same equivalences, and where they are not, the
        # spread is what exposes it
        for open_gap, extend_gap, matrix in (
            (-11, -1, "BLOSUM62"),
            (-4, -0.2, "BLOSUM62"),
            (-20, -3, "BLOSUM62"),
            (-11, -1, "BLOSUM45"),
            (-6, -0.5, "BLOSUM80"),
        ):
            aligned, _ = _pairwise_to_reference(
                records, reference, open_gap=open_gap, extend_gap=extend_gap, matrix=matrix
            )
            variants.append(aligned)
    return variants


def _equivalences(aligned: Dict[str, str], reference: str) -> Dict[Tuple[str, int], int]:
    """Map (species, reference residue index) -> that species' residue index.

    This is the thing a point mutant actually rests on: which residue of the
    non-binder corresponds to a given residue of the reference.
    """
    ref_aligned = aligned[reference]
    positions: Dict[str, List[int]] = {}
    for name, sequence in aligned.items():
        index = -1
        mapped = []
        for character in sequence:
            if character != GAP:
                index += 1
            mapped.append(index if character != GAP else -1)
        positions[name] = mapped

    out: Dict[Tuple[str, int], int] = {}
    ref_index = -1
    for column, character in enumerate(ref_aligned):
        if character == GAP:
            continue
        ref_index += 1
        for name in aligned:
            if name == reference:
                continue
            out[(name, ref_index)] = positions[name][column]
    return out


def alignment_confidence(
    alignment: Alignment,
    records: Sequence[SpeciesRecord],
    threads: int = 1,
) -> Tuple[Dict[int, float], List[str]]:
    """Per-reference-residue agreement across independently built alignments.

    1.0 means every method placed the same residue of every other species
    opposite this one. Low values mark loops where the region is right but the
    specific residue pairing is a guess - the case where a chimera is safe and a
    point mutant is not.
    """
    warnings_: List[str] = []
    try:
        variants = _alternative_alignments(records, alignment.reference, threads=threads)
    except Exception as exc:  # pragma: no cover - defensive
        return {}, [f"alignment confidence could not be computed: {exc}"]
    if not variants:
        return {}, ["alignment confidence could not be computed: no alternative alignment"]

    baseline = _equivalences(alignment.sequences, alignment.reference)
    others = [_equivalences(v, alignment.reference) for v in variants]
    n_reference = len(alignment.sequences[alignment.reference].replace(GAP, ""))
    species = [name for name in alignment.sequences if name != alignment.reference]

    confidence: Dict[int, float] = {}
    for ref_index in range(n_reference):
        agreements = []
        for name in species:
            expected = baseline.get((name, ref_index))
            for variant in others:
                agreements.append(1.0 if variant.get((name, ref_index)) == expected else 0.0)
        confidence[ref_index] = sum(agreements) / len(agreements) if agreements else 1.0

    method_note = (
        f"alignment confidence from {len(variants)} alternative alignment(s)"
        if any(shutil.which(tool) for tool in ("mafft", "muscle"))
        else "alignment confidence from re-scored pairwise alignments only "
        "(MAFFT/MUSCLE not installed), which understates the uncertainty"
    )
    warnings_.append(method_note)
    return confidence, warnings_


def local_identity(
    alignment: Alignment, window: int = IDENTITY_WINDOW
) -> Dict[int, float]:
    """Mean identity to the reference in a +/-``window`` sliding window.

    A loop where identity drops well below the rest of the chain is where the
    aligner has the least to go on.
    """
    reference = alignment.reference
    ref_aligned = alignment.sequences[reference]
    columns = [i for i, c in enumerate(ref_aligned) if c != GAP]
    others = [name for name in alignment.sequences if name != reference]

    per_position: List[float] = []
    for column in columns:
        matches = comparisons = 0
        for name in others:
            character = alignment.sequences[name][column]
            if character == GAP:
                comparisons += 1
                continue
            comparisons += 1
            matches += 1 if character == ref_aligned[column] else 0
        per_position.append(100.0 * matches / comparisons if comparisons else 0.0)

    out: Dict[int, float] = {}
    for index in range(len(per_position)):
        low = max(0, index - window)
        high = min(len(per_position), index + window + 1)
        chunk = per_position[low:high]
        out[index] = sum(chunk) / len(chunk) if chunk else 0.0
    return out
