"""Sequence input, UniProt fetching, and binding-call parsing."""

from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

BINDER = "binder"
NON_BINDER = "non_binder"
UNKNOWN = "unknown"
VALID_CALLS = (BINDER, NON_BINDER, UNKNOWN)

_UNIPROT_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)
_UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"

_ALIAS_NORMALISER = re.compile(r"[^a-z0-9]+")


class InputError(Exception):
    """Raised for malformed or inconsistent user input."""


@dataclass
class SpeciesRecord:
    """One species: its label, ungapped sequence, and binding call."""

    name: str
    sequence: str
    call: str = UNKNOWN
    source: str = "fasta"
    description: str = ""

    @property
    def is_binder(self) -> bool:
        return self.call == BINDER

    @property
    def is_non_binder(self) -> bool:
        return self.call == NON_BINDER

    @property
    def is_scored(self) -> bool:
        return self.call in (BINDER, NON_BINDER)


@dataclass
class Dataset:
    """All species records plus bookkeeping about how they were loaded."""

    records: List[SpeciesRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __iter__(self) -> Iterable[SpeciesRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def names(self) -> List[str]:
        return [r.name for r in self.records]

    @property
    def binders(self) -> List[SpeciesRecord]:
        return [r for r in self.records if r.is_binder]

    @property
    def non_binders(self) -> List[SpeciesRecord]:
        return [r for r in self.records if r.is_non_binder]

    @property
    def scored(self) -> List[SpeciesRecord]:
        return [r for r in self.records if r.is_scored]

    def get(self, name: str) -> SpeciesRecord:
        for r in self.records:
            if r.name == name:
                return r
        match = _resolve_alias(name, self.names)
        if match is None:
            raise InputError(
                f"species {name!r} not found; known species: {', '.join(self.names)}"
            )
        return self.get(match)


def _normalise(label: str) -> str:
    return _ALIAS_NORMALISER.sub("", label.strip().lower())


def _resolve_alias(label: str, candidates: Sequence[str]) -> Optional[str]:
    """Match a user-supplied species label against FASTA identifiers.

    Exact match wins, then case/punctuation-insensitive match, then a unique
    prefix or substring match (so ``mouse`` finds ``mouse_CD73|Q61503``).
    """
    if label in candidates:
        return label
    target = _normalise(label)
    norm = {c: _normalise(c) for c in candidates}
    exact = [c for c, n in norm.items() if n == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise InputError(f"species label {label!r} is ambiguous: {exact}")
    partial = [c for c, n in norm.items() if n.startswith(target) or target in n]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise InputError(
            f"species label {label!r} matches several sequences: {partial}. "
            "Use the full FASTA identifier."
        )
    return None


def parse_fasta(text: str) -> List[SpeciesRecord]:
    """Minimal FASTA reader (no Biopython dependency for plain text input)."""
    records: List[SpeciesRecord] = []
    name: Optional[str] = None
    description = ""
    chunks: List[str] = []

    def flush() -> None:
        if name is not None:
            seq = "".join(chunks).replace(" ", "").replace("\r", "").upper()
            seq = seq.replace("*", "").replace("-", "")
            if not seq:
                raise InputError(f"FASTA record {name!r} has no sequence")
            records.append(SpeciesRecord(name=name, sequence=seq, description=description))

    for line in text.splitlines():
        if line.startswith(">"):
            flush()
            header = line[1:].strip()
            name = header.split()[0] if header else ""
            description = header
            chunks = []
            if not name:
                raise InputError("FASTA record with an empty identifier")
        else:
            chunks.append(line.strip())
    flush()
    if not records:
        raise InputError("no FASTA records found")
    seen = set()
    for r in records:
        if r.name in seen:
            raise InputError(f"duplicate FASTA identifier {r.name!r}")
        seen.add(r.name)
    return records


def _fetch_uniprot(accession: str, cache_dir: Optional[Path] = None) -> SpeciesRecord:
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{accession}.fasta"
        if cache_file.exists():
            return parse_fasta(cache_file.read_text())[0]
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a declared dep
        raise InputError(
            f"cannot fetch UniProt accession {accession}: requests is not installed"
        ) from exc
    url = _UNIPROT_FASTA_URL.format(acc=accession)
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise InputError(
            f"UniProt fetch failed for {accession}: HTTP {response.status_code}"
        )
    if cache_file is not None:
        cache_file.write_text(response.text)
    record = parse_fasta(response.text)[0]
    record.source = "uniprot"
    return record


def load_sequences(
    spec: Sequence[str] | str | os.PathLike, cache_dir: Optional[Path] = None
) -> List[SpeciesRecord]:
    """Load sequences from a FASTA path, or fetch a list of UniProt accessions.

    ``spec`` may be a path to a FASTA file, or a sequence of items each of which
    is either a UniProt accession (``Q61503``) or ``label=ACCESSION`` to give the
    fetched sequence a friendlier species name.
    """
    if isinstance(spec, (str, os.PathLike)) and Path(spec).exists():
        return parse_fasta(Path(spec).read_text())

    items: List[str]
    if isinstance(spec, (str, os.PathLike)):
        items = [s for s in re.split(r"[,\s]+", str(spec)) if s]
    else:
        items = list(spec)
    if not items:
        raise InputError("no sequences supplied")

    records: List[SpeciesRecord] = []
    for item in items:
        label: Optional[str] = None
        accession = item
        if "=" in item:
            label, accession = item.split("=", 1)
        accession = accession.strip()
        if not _UNIPROT_RE.match(accession):
            raise InputError(
                f"{item!r} is neither an existing FASTA path nor a UniProt accession"
            )
        record = _fetch_uniprot(accession, cache_dir=cache_dir)
        if label:
            record.name = label.strip()
        records.append(record)
    return records


def _read_binding_rows(path: Path) -> List[Dict[str, str]]:
    text = Path(path).read_text()
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise InputError(
                "PyYAML is required to read YAML binding files; use CSV instead"
            ) from exc
        data = yaml.safe_load(text)
        rows: List[Dict[str, str]] = []
        if isinstance(data, dict):
            # {species: call} or {binding: {species: call}}
            if "binding" in data and isinstance(data["binding"], dict):
                data = data["binding"]
            for species, call in data.items():
                rows.append({"species": str(species), "binding": str(call)})
        elif isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    raise InputError("YAML binding list entries must be mappings")
                rows.append({str(k): str(v) for k, v in entry.items()})
        else:
            raise InputError("unrecognised YAML binding-call structure")
        return rows

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise InputError(f"{path} is empty")
    return [{(k or "").strip(): (v or "").strip() for k, v in row.items()} for row in reader]


_CALL_SYNONYMS = {
    "binder": BINDER,
    "binding": BINDER,
    "bind": BINDER,
    "yes": BINDER,
    "positive": BINDER,
    "pos": BINDER,
    "1": BINDER,
    "true": BINDER,
    "non_binder": NON_BINDER,
    "nonbinder": NON_BINDER,
    "non-binder": NON_BINDER,
    "no": NON_BINDER,
    "negative": NON_BINDER,
    "neg": NON_BINDER,
    "0": NON_BINDER,
    "false": NON_BINDER,
    "unknown": UNKNOWN,
    "na": UNKNOWN,
    "n/a": UNKNOWN,
    "": UNKNOWN,
}


def normalise_call(value: str) -> str:
    call = _CALL_SYNONYMS.get(str(value).strip().lower())
    if call is None:
        raise InputError(
            f"binding call {value!r} not understood; use one of {VALID_CALLS}"
        )
    return call


def load_binding_calls(path: Path) -> Dict[str, str]:
    """Parse a CSV/YAML binding table into ``{species: call}``."""
    rows = _read_binding_rows(Path(path))
    calls: Dict[str, str] = {}
    for row in rows:
        keys = {k.lower(): k for k in row}
        species_key = keys.get("species") or keys.get("name") or keys.get("organism")
        call_key = keys.get("binding") or keys.get("call") or keys.get("binds")
        if species_key is None or call_key is None:
            raise InputError(
                "binding file must have 'species' and 'binding' columns; "
                f"found {sorted(row)}"
            )
        species = row[species_key].strip()
        if not species:
            continue
        if species in calls:
            raise InputError(f"duplicate binding row for {species!r}")
        calls[species] = normalise_call(row[call_key])
    if not calls:
        raise InputError(f"no binding calls found in {path}")
    return calls


def build_dataset(
    records: Sequence[SpeciesRecord], calls: Dict[str, str], reference: str
) -> "tuple[Dataset, str]":
    """Attach binding calls to sequences and validate the combination.

    Returns the populated dataset and the resolved reference sequence name.
    """
    dataset = Dataset(records=list(records))
    names = dataset.names
    assigned: Dict[str, str] = {}
    for label, call in calls.items():
        match = _resolve_alias(label, names)
        if match is None:
            raise InputError(
                f"binding call for {label!r} has no matching sequence "
                f"(sequences: {', '.join(names)})"
            )
        if match in assigned:
            raise InputError(f"two binding calls map onto sequence {match!r}")
        assigned[match] = call
    for record in dataset.records:
        if record.name in assigned:
            record.call = assigned[record.name]
        else:
            record.call = UNKNOWN
            dataset.warnings.append(
                f"no binding call for {record.name!r}; treated as 'unknown' "
                "(aligned but excluded from scoring)"
            )

    ref_match = _resolve_alias(reference, names)
    if ref_match is None:
        raise InputError(
            f"reference species {reference!r} not among sequences: {', '.join(names)}"
        )
    ref_record = dataset.get(ref_match)
    if not ref_record.is_binder:
        raise InputError(
            f"reference species {ref_match!r} is marked {ref_record.call!r}; "
            "the reference must be a binder (numbering and structure are anchored to it)"
        )

    if len(dataset) < 3:
        dataset.warnings.append(
            f"only {len(dataset)} sequences supplied; 6-8 species gives far more "
            "resolving power than the 3-species minimum"
        )
    if not dataset.binders:
        raise InputError("no binder species in the binding table")
    if not dataset.non_binders:
        raise InputError("no non_binder species in the binding table")
    return dataset, ref_match
