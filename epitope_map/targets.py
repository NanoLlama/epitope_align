"""Built-in target profiles: defaults for a protein you analyse repeatedly.

A profile is a data file under ``data/targets/``, not constants in the code, so
adding a target is adding a file. Everything a profile supplies is a *default*:
an explicitly passed flag always wins, and :attr:`TargetProfile.provenance`
records which values came from where so the report can say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .io_seq import InputError

TARGET_DIR = Path(__file__).parent / "data" / "targets"


@dataclass
class AvoidEntry:
    accession: str
    species: str
    reason: str


@dataclass
class TargetProfile:
    """Defaults for one target, loaded from a YAML file."""

    key: str
    name: str = ""
    description: str = ""
    reference: Optional[str] = None
    sequences: Dict[str, str] = field(default_factory=dict)
    domains: List[Dict[str, Any]] = field(default_factory=list)
    domain_aliases: Dict[str, List[str]] = field(default_factory=dict)
    topology: Dict[str, Sequence[int]] = field(default_factory=dict)
    oligomer: str = ""
    oligomer_interface_domains: List[str] = field(default_factory=list)
    structures: Dict[str, str] = field(default_factory=dict)
    candidate_species: Dict[str, str] = field(default_factory=dict)
    avoid: List[AvoidEntry] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    identity_guards: List[Dict[str, Any]] = field(default_factory=list)
    path: Optional[Path] = None

    # ---- rendering into CLI-shaped values -------------------------------
    def sequences_value(self) -> str:
        return ",".join(f"{label}={acc}" for label, acc in self.sequences.items())

    def candidates_value(self) -> List[str]:
        return [f"{label}={acc}" for label, acc in self.candidate_species.items()]

    def topology_value(self) -> str:
        parts = []
        for kind in ("cytoplasmic", "transmembrane", "extracellular"):
            span = self.topology.get(kind)
            if span:
                parts.append(f"{kind}={int(span[0])}-{int(span[1])}")
        return ",".join(parts)

    def domains_table(self) -> str:
        lines = ["name\tstart\tend"]
        for domain in self.domains:
            lines.append(
                f"{domain['name']}\t{int(domain['start'])}\t{int(domain['end'])}"
            )
        return "\n".join(lines) + "\n"

    def interface_ranges(self) -> List[Tuple[int, int]]:
        """Ranges of the domains that form the oligomer interface."""
        wanted = set(self.oligomer_interface_domains)
        return [
            (int(d["start"]), int(d["end"]))
            for d in self.domains
            if d.get("name") in wanted
        ]

    def avoided(self, accession: str) -> Optional[AvoidEntry]:
        for entry in self.avoid:
            if entry.accession and entry.accession.upper() == accession.upper():
                return entry
        return None

    def structure_recommendation(self) -> str:
        if not self.structures:
            return ""
        listed = ", ".join(f"{name} ({pdb})" for name, pdb in self.structures.items())
        return (
            f"experimental structures exist for this target: {listed}. They give a "
            "real biological assembly for SASA and a superposition target for "
            "--equivalence structural; check assembly and resolution before "
            "choosing one, which is why none is selected automatically"
        )


def _entries(payload: Any) -> List[Any]:
    return list(payload) if isinstance(payload, list) else []


def load_profile(key: str, directory: Optional[Path] = None) -> TargetProfile:
    """Load a profile by key, e.g. ``tfr1``."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dep
        raise InputError("PyYAML is required for --target") from exc

    directory = Path(directory or TARGET_DIR)
    path = directory / f"{key.strip().lower()}.yaml"
    if not path.exists():
        available = ", ".join(available_targets(directory)) or "none"
        raise InputError(
            f"unknown target {key!r}. Available profiles: {available}. "
            "Add one by dropping a YAML file into epitope_map/data/targets/."
        )
    payload = yaml.safe_load(path.read_text()) or {}

    avoid: List[AvoidEntry] = []
    notes: List[str] = []
    for entry in _entries(payload.get("avoid")):
        if not isinstance(entry, dict):
            continue
        if entry.get("note"):
            notes.append(str(entry["note"]))
            continue
        avoid.append(
            AvoidEntry(
                accession=str(entry.get("accession", "")),
                species=str(entry.get("species", "")),
                reason=str(entry.get("reason", "")).strip(),
            )
        )
    for guard in _entries(payload.get("identity_guards")):
        if isinstance(guard, dict) and guard.get("note"):
            notes.append(str(guard["note"]).strip())

    return TargetProfile(
        key=key.strip().lower(),
        name=str(payload.get("name", key)),
        description=str(payload.get("description", "")),
        reference=payload.get("reference_default"),
        sequences={str(k): str(v) for k, v in (payload.get("sequences") or {}).items()},
        domains=[dict(d) for d in _entries(payload.get("domains"))],
        domain_aliases={
            str(k): [str(v) for v in values]
            for k, values in (payload.get("domain_aliases") or {}).items()
        },
        topology={str(k): list(v) for k, v in (payload.get("topology") or {}).items()},
        oligomer=str(payload.get("oligomer", "")),
        oligomer_interface_domains=[
            str(v) for v in _entries(payload.get("oligomer_interface_domains"))
        ],
        structures={
            str(k): str(v) for k, v in (payload.get("structures") or {}).items()
        },
        candidate_species={
            str(k): str(v)
            for k, v in (payload.get("candidate_species") or {}).items()
        },
        avoid=avoid,
        notes=notes,
        identity_guards=[
            dict(g) for g in _entries(payload.get("identity_guards")) if isinstance(g, dict)
        ],
        path=path,
    )


def available_targets(directory: Optional[Path] = None) -> List[str]:
    directory = Path(directory or TARGET_DIR)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def describe_targets(directory: Optional[Path] = None) -> str:
    """One line per available profile, for ``--list-targets``."""
    keys = available_targets(directory)
    if not keys:
        return "No target profiles are installed."
    lines = ["Available target profiles:", ""]
    for key in keys:
        try:
            profile = load_profile(key, directory)
        except InputError:  # pragma: no cover - defensive
            continue
        lines.append(f"  {key:12s} {profile.name}")
        if profile.description:
            lines.append(f"  {'':12s} {profile.description}")
        if profile.sequences:
            lines.append(
                f"  {'':12s} species: {', '.join(profile.sequences)}"
            )
        lines.append("")
    lines.append("Use with:  epitope-map --target <key> --binding binding.csv")
    return "\n".join(lines)
