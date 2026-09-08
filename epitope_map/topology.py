"""Membrane topology and domain annotation.

An antibody reaches the outside of a cell and nothing else. Without that
constraint the pipeline will happily rank a cytoplasmic patch first - it has no
way to know the difference - and a cytoplasmic N-X-S/T is never glycosylated
however good it looks in sequence. Both failures were seen on a real target, so
topology is a first-class input here rather than an optional filter.

Ranges in this module are in **reference sequence numbering** (1-based over the
ungapped reference sequence), which is what UniProt features use. Conversion to
structure author numbering is :class:`epitope_map.align.ResidueMap`'s job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .io_seq import InputError

#: Sequons this close to the extracellular boundary have unreliable occupancy.
BOUNDARY_MARGIN = 5

EXTRACELLULAR = "extracellular"
TRANSMEMBRANE = "transmembrane"
CYTOPLASMIC = "cytoplasmic"
SIGNAL = "signal_peptide"
UNKNOWN_TOPOLOGY = "unknown"

_ALIASES = {
    "extracellular": EXTRACELLULAR,
    "extra": EXTRACELLULAR,
    "ecd": EXTRACELLULAR,
    "ectodomain": EXTRACELLULAR,
    "lumenal": EXTRACELLULAR,
    "luminal": EXTRACELLULAR,
    "periplasmic": EXTRACELLULAR,
    "tm": TRANSMEMBRANE,
    "transmembrane": TRANSMEMBRANE,
    "helical": TRANSMEMBRANE,
    "cyto": CYTOPLASMIC,
    "cytoplasmic": CYTOPLASMIC,
    "intracellular": CYTOPLASMIC,
    "signal": SIGNAL,
    "signal_peptide": SIGNAL,
}


@dataclass
class Segment:
    """A labelled stretch of the reference sequence, 1-based inclusive."""

    kind: str
    start: int
    end: int
    description: str = ""

    def contains(self, position: int) -> bool:
        return self.start <= position <= self.end

    @property
    def label(self) -> str:
        return f"{self.kind} {self.start}-{self.end}"


@dataclass
class Topology:
    """Where each part of the reference chain sits relative to the membrane."""

    segments: List[Segment] = field(default_factory=list)
    source: str = "none"
    length: Optional[int] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return bool(self.segments)

    @property
    def whole_chain(self) -> bool:
        """True when the caller explicitly said the whole chain is accessible."""
        return self.source == "whole-chain"

    def kind_at(self, position: int) -> str:
        for segment in self.segments:
            if segment.contains(position):
                return segment.kind
        return UNKNOWN_TOPOLOGY

    def is_accessible(self, position: int) -> bool:
        """Could an antibody in the extracellular space reach this residue?"""
        if self.whole_chain or not self.known:
            return True
        return self.kind_at(position) == EXTRACELLULAR

    def extracellular_ranges(self) -> List[Tuple[int, int]]:
        return [(s.start, s.end) for s in self.segments if s.kind == EXTRACELLULAR]

    def near_boundary(self, position: int, margin: int = BOUNDARY_MARGIN) -> bool:
        """Within ``margin`` residues of the edge of an extracellular stretch."""
        for start, end in self.extracellular_ranges():
            if start <= position <= end and (
                position - start < margin or end - position < margin
            ):
                return True
        return False

    def summary(self) -> str:
        if self.whole_chain:
            return "whole chain treated as accessible (declared by the user)"
        if not self.known:
            return "unknown"
        parts = [f"{s.kind} {s.start}-{s.end}" for s in self.segments]
        return f"{', '.join(parts)} (from {self.source})"


def parse_topology_spec(spec: str, length: Optional[int] = None) -> Topology:
    """Parse ``extracellular=121-763,tm=68-88,cytoplasmic=1-67``.

    The literal ``whole-chain`` declares that the entire chain is accessible,
    which is the explicit way to run a soluble protein or a pre-trimmed
    ectodomain construct.
    """
    text = str(spec).strip()
    if not text:
        raise InputError("empty --topology")
    if text.lower().replace("_", "-") in ("whole-chain", "wholechain", "all", "none"):
        return Topology(source="whole-chain", length=length)

    segments: List[Segment] = []
    for chunk in re.split(r"[,;]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk and ":" not in chunk:
            raise InputError(
                f"cannot read {chunk!r} in --topology: expected e.g. "
                "'extracellular=121-763'"
            )
        key, _, value = re.split(r"([=:])", chunk, maxsplit=1)[0], "", ""
        key, value = re.split(r"[=:]", chunk, maxsplit=1)
        kind = _ALIASES.get(key.strip().lower().replace("-", "_"))
        if kind is None:
            raise InputError(
                f"unknown topology label {key.strip()!r}; use one of "
                "extracellular, transmembrane, cytoplasmic, signal"
            )
        for span in re.split(r"[|+]", value):
            span = span.strip()
            if not span:
                continue
            bounds = re.split(r"\s*(?:-|\.\.)\s*", span)
            if len(bounds) != 2:
                raise InputError(f"cannot read range {span!r} in --topology")
            try:
                start, end = int(bounds[0]), int(bounds[1])
            except ValueError:
                raise InputError(f"range bounds must be integers: {span!r}") from None
            if start > end:
                raise InputError(f"range start after end: {span!r}")
            segments.append(Segment(kind=kind, start=start, end=end))

    if not segments:
        raise InputError(f"no ranges found in --topology {spec!r}")
    topology = Topology(segments=sorted(segments, key=lambda s: s.start),
                        source="--topology", length=length)
    if not topology.extracellular_ranges():
        topology.warnings.append(
            "--topology declares no extracellular range, so no residue is "
            "reachable by an antibody; did you mean to name one?"
        )
    return topology


def topology_from_range(start: int, end: int, source: str = "--ectodomain") -> Topology:
    """Treat an explicit ectodomain range as the extracellular segment."""
    return Topology(
        segments=[Segment(kind=EXTRACELLULAR, start=start, end=end)], source=source
    )


# --------------------------------------------------------------------------
# UniProt features
# --------------------------------------------------------------------------

_FEATURE_KIND = {
    "topological domain": "topo_dom",
    "topo_dom": "topo_dom",
    "transmembrane": "transmem",
    "transmem": "transmem",
    "intramembrane": "intramem",
    "signal peptide": "signal",
    "signal": "signal",
    "domain": "domain",
    "region": "region",
    "repeat": "domain",
}


def parse_uniprot_features(payload: dict) -> Tuple[Topology, List[Segment]]:
    """Derive topology and domain annotation from a UniProt JSON entry.

    Returns ``(topology, domains)``. An entry with no topological features gives
    an unknown topology - which the pipeline treats as a reason to stop and ask,
    not as permission to score the whole chain.
    """
    length = None
    try:
        length = int(payload.get("sequence", {}).get("length"))
    except (TypeError, ValueError):
        length = None

    segments: List[Segment] = []
    domains: List[Segment] = []
    for feature in payload.get("features", []) or []:
        kind = _FEATURE_KIND.get(str(feature.get("type", "")).strip().lower())
        if kind is None:
            continue
        location = feature.get("location", {}) or {}
        try:
            start = int(location.get("start", {}).get("value"))
            end = int(location.get("end", {}).get("value"))
        except (TypeError, ValueError):
            continue
        description = str(feature.get("description", "") or "").strip()

        if kind == "topo_dom":
            mapped = _ALIASES.get(description.split(";")[0].strip().lower())
            segments.append(
                Segment(
                    kind=mapped or UNKNOWN_TOPOLOGY,
                    start=start,
                    end=end,
                    description=description,
                )
            )
        elif kind in ("transmem", "intramem"):
            segments.append(
                Segment(kind=TRANSMEMBRANE, start=start, end=end, description=description)
            )
        elif kind == "signal":
            segments.append(
                Segment(kind=SIGNAL, start=start, end=end, description=description)
            )
        elif kind in ("domain", "region"):
            domains.append(
                Segment(kind=kind, start=start, end=end, description=description or kind)
            )

    topology = Topology(
        segments=sorted(segments, key=lambda s: s.start),
        source="UniProt features" if segments else "none",
        length=length,
    )
    if segments and not topology.extracellular_ranges():
        topology.warnings.append(
            "UniProt lists topological features for the reference but no "
            "extracellular segment; check that this is a membrane protein and "
            "supply --topology if not"
        )
    return topology, sorted(domains, key=lambda s: s.start)


def load_domains_tsv(path: Path) -> List[Segment]:
    """Read a manual domain table: ``name<TAB>start<TAB>end`` (header optional)."""
    segments: List[Segment] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = re.split(r"[\t,]", line)
        if len(fields) < 3:
            raise InputError(
                f"{path}:{line_number}: expected 'name<TAB>start<TAB>end'"
            )
        name = fields[0].strip()
        if name.lower() in ("name", "domain"):
            continue  # header
        try:
            start, end = int(fields[1]), int(fields[2])
        except ValueError:
            raise InputError(f"{path}:{line_number}: start and end must be integers") from None
        segments.append(Segment(kind="domain", start=start, end=end, description=name))
    if not segments:
        raise InputError(f"no domains found in {path}")
    return sorted(segments, key=lambda s: s.start)


def domain_at(domains: Sequence[Segment], position: int) -> Optional[Segment]:
    """Smallest annotated domain covering a position, if any."""
    covering = [d for d in domains if d.contains(position)]
    if not covering:
        return None
    return min(covering, key=lambda d: d.end - d.start)


class TopologyUnknown(InputError):
    """Raised when nothing says which part of the chain faces the outside."""


def require_topology(
    topology: Optional[Topology], chain_length: int, reference: str
) -> Topology:
    """Refuse to run rather than silently scoring the inside of a cell."""
    if topology is not None and (topology.known or topology.whole_chain):
        return topology
    raise TopologyUnknown(
        f"the membrane topology of the reference ({reference}, {chain_length} "
        "residues) could not be determined, and an antibody can only reach the "
        "extracellular part.\n"
        "Supply one of:\n"
        "  --topology extracellular=121-763,tm=68-88,cytoplasmic=1-67\n"
        "  --ectodomain 121-763        (treated as the extracellular range)\n"
        "  --topology whole-chain      (say so explicitly for a soluble protein "
        "or an ectodomain-only construct)\n"
        "Fetching the sequences from UniProt accessions lets the topology be "
        "read from the entry's TOPO_DOM/TRANSMEM features automatically."
    )
