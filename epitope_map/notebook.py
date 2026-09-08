"""Helpers for the Colab notebook, where every input arrives as one line.

Colab's form fields are single-line text boxes: a user cannot press Enter inside
one. So the binding table, which is naturally one row per species, has to be
typed on a single line - and people will reasonably reach for any of half a
dozen separators. Everything here is about accepting what they actually type and
telling them plainly when it cannot be read.

Kept in the package rather than in the notebook so it is covered by the tests.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .io_seq import InputError, normalise_call, parse_fasta

_ENTRY_SEPARATORS = re.compile(r"[;\n]+")
_LABEL_CALL_SEPARATORS = re.compile(r"\s*[=:]\s*")


def parse_binding_calls(text: str) -> Dict[str, str]:
    """Read a binding table typed on one line into ``{species: call}``.

    All of these mean the same thing::

        mouse=binder, rat=binder, human=non_binder
        mouse: binder; rat: binder; human: non_binder
        mouse,binder; rat,binder; human,non_binder
        mouse,binder,rat,binder,human,non_binder
        species,binding\\nmouse,binder\\nrat,binder\\nhuman,non_binder

    The call itself may be any spelling :func:`epitope_map.io_seq.normalise_call`
    understands (``binder``/``yes``/``1``, ``non_binder``/``no``/``0``,
    ``unknown``).
    """
    if not str(text).strip():
        raise InputError("no binding calls given")

    # a literal backslash-n typed into a form box means a line break
    normalised = str(text).replace("\\n", "\n")
    calls: Dict[str, str] = {}

    for line in _ENTRY_SEPARATORS.split(normalised):
        line = line.strip().strip(",")
        if not line:
            continue
        if _LABEL_CALL_SEPARATORS.search(line):
            entries = [part for part in line.split(",") if part.strip()]
        else:
            fields = [field.strip() for field in line.split(",") if field.strip()]
            if len(fields) == 1:
                raise InputError(
                    f"cannot read {line.strip()!r} as a binding call: expected "
                    "something like 'mouse=binder' or 'mouse,binder'"
                )
            if fields[0].lower() in ("species", "name", "organism") and len(fields) == 2:
                continue  # a pasted CSV header
            if len(fields) % 2:
                raise InputError(
                    f"cannot read {line.strip()!r}: it has an odd number of "
                    "comma-separated values, so the species and calls do not pair up"
                )
            entries = [
                f"{fields[i]}={fields[i + 1]}" for i in range(0, len(fields), 2)
            ]

        for entry in entries:
            parts = _LABEL_CALL_SEPARATORS.split(entry.strip(), maxsplit=1)
            if len(parts) != 2:
                parts = [p.strip() for p in entry.split(",", 1)]
            if len(parts) != 2 or not parts[0].strip():
                raise InputError(
                    f"cannot read {entry.strip()!r} as 'species=call'"
                )
            species, call = parts[0].strip(), parts[1].strip()
            if species.lower() in ("species", "name", "organism"):
                continue
            if species in calls:
                raise InputError(f"{species!r} is listed twice in the binding table")
            calls[species] = normalise_call(call)

    if not calls:
        raise InputError("no binding calls could be read")
    return calls


def write_binding_csv(calls: Dict[str, str], path: Path) -> Path:
    """Write parsed calls out in the CSV form the CLI reads."""
    path = Path(path)
    path.write_text(
        "species,binding\n" + "".join(f"{s},{c}\n" for s, c in calls.items())
    )
    return path


def sequence_labels(spec: str) -> List[str]:
    """Species labels implied by a ``--sequences`` value (accessions or FASTA)."""
    spec = str(spec).strip()
    if Path(spec).exists():
        return [record.name for record in parse_fasta(Path(spec).read_text())]
    labels = []
    for item in re.split(r"[,\s]+", spec):
        if not item:
            continue
        labels.append(item.split("=", 1)[0].strip() if "=" in item else item.strip())
    return labels


def check_panel(
    calls: Dict[str, str],
    labels: Sequence[str],
    reference: str,
) -> Tuple[List[str], List[str]]:
    """Catch the mistakes worth catching before a run, not during it.

    Returns ``(problems, notes)``: problems must be fixed, notes are worth
    reading but do not stop a run.
    """
    problems: List[str] = []
    notes: List[str] = []

    def norm(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    label_by_norm = {norm(label): label for label in labels}
    for species in calls:
        if norm(species) not in label_by_norm:
            problems.append(
                f"'{species}' has a binding result but no sequence. Sequence "
                f"labels are: {', '.join(labels)}"
            )
    for label in labels:
        if norm(label) not in {norm(s) for s in calls}:
            notes.append(
                f"'{label}' has a sequence but no binding result, so it will be "
                "aligned and shown but not used for scoring"
            )

    binders = [s for s, c in calls.items() if c == "binder"]
    non_binders = [s for s, c in calls.items() if c == "non_binder"]
    if not binders:
        problems.append("no species is marked 'binder' - at least one is needed")
    if not non_binders:
        problems.append(
            "no species is marked 'non_binder' - the whole method rests on the "
            "contrast, so at least one is needed"
        )

    if norm(reference) not in {norm(s) for s in calls}:
        problems.append(
            f"the reference '{reference}' is not in the binding table"
        )
    elif calls[
        next(s for s in calls if norm(s) == norm(reference))
    ] != "binder":
        problems.append(
            f"the reference '{reference}' must be a species the antibody binds - "
            "all numbering in the results refers to it"
        )

    scored = len(binders) + len(non_binders)
    if scored < 3:
        notes.append(
            f"only {scored} species will be scored. Three is the minimum and six "
            "to eight is far better - each extra informative species roughly "
            "halves the candidate list"
        )
    return problems, notes


def summarise(calls: Dict[str, str], reference: str) -> str:
    """A short table to print back, so the user can see what was understood."""
    width = max([len(s) for s in calls] + [len("species")])
    lines = [f"{'species'.ljust(width)}  binding"]
    lines.append(f"{'-' * width}  {'-' * 10}")
    for species, call in calls.items():
        marker = "  <- reference" if species.lower() == reference.lower().strip() else ""
        lines.append(f"{species.ljust(width)}  {call}{marker}")
    return "\n".join(lines)
