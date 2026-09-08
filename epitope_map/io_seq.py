"""Sequence input, UniProt fetching, and binding-call parsing."""

from __future__ import annotations

import csv
import io
import json
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
_UNIPROT_JSON_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"

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
    accession: str = ""
    #: Raw UniProt entry, when the sequence was fetched by accession. Carries the
    #: TOPO_DOM/TRANSMEM/DOMAIN features the topology and domain annotation are
    #: derived from, so a fetched reference needs no hand-written ranges.
    features: Optional[dict] = None

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


def _uniprot_get(url: str, timeout: int = 60):
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - requests is a declared dep
        raise InputError(
            f"cannot fetch {url}: requests is not installed"
        ) from exc
    try:
        return requests.get(url, timeout=timeout)
    except Exception as exc:  # network, proxy, DNS, TLS
        # a blocked or offline network is a normal condition for this tool, not
        # a crash: say which accession could not be reached and why
        raise InputError(
            f"could not reach UniProt for {url.rsplit('/', 1)[-1]}: {exc.__class__.__name__}. "
            "If this machine has no outbound access, download the sequences and "
            "pass a FASTA file instead."
        ) from None


def _fetch_uniprot(accession: str, cache_dir: Optional[Path] = None) -> SpeciesRecord:
    """Fetch a sequence and, alongside it, the entry's features.

    The features are what make topology automatic; they are cached next to the
    FASTA so a repeated run needs no network.
    """
    fasta_cache = json_cache = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fasta_cache = cache_dir / f"{accession}.fasta"
        json_cache = cache_dir / f"{accession}.json"

    if fasta_cache is not None and fasta_cache.exists():
        record = parse_fasta(fasta_cache.read_text())[0]
    else:
        response = _uniprot_get(_UNIPROT_FASTA_URL.format(acc=accession))
        if response.status_code != 200:
            raise InputError(
                f"UniProt fetch failed for {accession}: HTTP "
                f"{response.status_code}"
                + (
                    " - no such accession"
                    if response.status_code == 404
                    else ""
                )
            )
        if fasta_cache is not None:
            fasta_cache.write_text(response.text)
        record = parse_fasta(response.text)[0]

    record.source = "uniprot"
    record.accession = accession
    record.features = _fetch_uniprot_features(accession, json_cache)
    return record


def _fetch_uniprot_features(
    accession: str, cache_file: Optional[Path]
) -> Optional[dict]:
    """The entry as JSON, or ``None`` if it could not be had.

    Failure here is not fatal: without features the pipeline asks the user for
    the topology instead of guessing.
    """
    if cache_file is not None and cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except json.JSONDecodeError:
            pass
    try:
        response = _uniprot_get(_UNIPROT_JSON_URL.format(acc=accession))
    except InputError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if cache_file is not None:
        cache_file.write_text(json.dumps(payload))
    return payload


ACCEPTED_FORMATS = (
    "  a FASTA file path            candidates.fasta\n"
    "  label=ACCESSION, comma list  dog=Q9GLD3,cat=Q9MYZ3\n"
    "  bare accessions, comma list  Q9GLD3,Q9MYZ3\n"
    "  any mixture of the above, and the flag may be repeated"
)


def _find_nearby(name: str, root: Path = Path("."), limit: int = 3) -> List[Path]:
    """Somewhere else under the working directory, is there a file of this name?

    Uploading to one directory and pointing at another is the single most common
    way this goes wrong, so it is worth answering rather than just denying.
    """
    try:
        return sorted(root.rglob(name))[:limit]
    except (OSError, ValueError):  # pragma: no cover - defensive
        return []


def _describe_missing_path(token: str) -> str:
    path = Path(token)
    message = f"no such file: {token}"
    nearby = _find_nearby(path.name)
    if nearby:
        message += (
            "\n  a file of that name does exist at: "
            + ", ".join(str(p) for p in nearby)
        )
    parent = path.parent
    if str(parent) not in (".", "") and not parent.exists():
        message += f"\n  the directory {parent} does not exist either"
    return message


def _looks_like_path(token: str) -> bool:
    return (
        "/" in token
        or token.startswith("~")
        or token.lower().endswith((".fasta", ".fa", ".faa", ".fas", ".txt", ".seq"))
    )


def load_sequences(
    spec: Sequence[str] | str | os.PathLike, cache_dir: Optional[Path] = None
) -> List[SpeciesRecord]:
    """Load sequences from FASTA files, UniProt accessions, or a mixture.

    Every accepted shape is listed in :data:`ACCEPTED_FORMATS`. Failures name
    the item that failed and why, rather than reporting the whole argument as
    unrecognised - which told a user nothing when one accession in a list of
    nine was wrong, or when a file was uploaded to a different directory.
    """
    if isinstance(spec, (str, os.PathLike)) and Path(spec).exists():
        return parse_fasta(Path(spec).read_text())

    items: List[str] = []
    raw = [spec] if isinstance(spec, (str, os.PathLike)) else list(spec)
    for entry in raw:
        if isinstance(entry, (str, os.PathLike)) and Path(entry).exists():
            items.append(str(entry))  # a path may contain commas; do not split it
            continue
        # a repeated flag gives one list element per use, and each of those may
        # itself be a comma-separated list
        items.extend(part for part in re.split(r"[,\s]+", str(entry)) if part)
    if not items:
        raise InputError("no sequences supplied")

    records: List[SpeciesRecord] = []
    for item in items:
        label: Optional[str] = None
        accession = item
        if "=" in item:
            label, accession = item.split("=", 1)
        accession = accession.strip()

        # an item may itself be a FASTA file, so a list can mix files and
        # accessions - which is what a candidate-ortholog list tends to look like
        if Path(accession).exists():
            try:
                parsed = parse_fasta(Path(accession).read_text())
            except InputError as exc:
                raise InputError(
                    f"{accession} is not readable as FASTA: {exc}"
                ) from None
            except OSError as exc:
                raise InputError(f"cannot read {accession}: {exc}") from None
            if label and len(parsed) == 1:
                parsed[0].name = label.strip()
            records.extend(parsed)
            continue

        if _looks_like_path(accession):
            raise InputError(_describe_missing_path(accession))

        if not _UNIPROT_RE.match(accession):
            raise InputError(
                f"{item!r} is not a UniProt accession and does not look like a "
                f"file path. Accepted formats:\n{ACCEPTED_FORMATS}"
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
