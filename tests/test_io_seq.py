"""Sequence, binding-call parsing and dataset validation."""

import pytest

from epitope_map.io_seq import (
    BINDER,
    NON_BINDER,
    UNKNOWN,
    InputError,
    build_dataset,
    load_binding_calls,
    normalise_call,
    parse_fasta,
)

FASTA = """>mouse mouse ortholog
ACDEFGHIKL
MNPQRSTVWY
>rat
ACDEFGHIKLMNPQRSTVWY
>human
ACDEFGHIKLMNPQRSTVWA
>marmoset
ACDEFGHIKLMNPQRSTVWA
"""


def records():
    return parse_fasta(FASTA)


def test_fasta_parsing_joins_wrapped_lines():
    parsed = records()
    assert [r.name for r in parsed] == ["mouse", "rat", "human", "marmoset"]
    assert parsed[0].sequence == "ACDEFGHIKLMNPQRSTVWY"
    assert parsed[0].description == "mouse mouse ortholog"


def test_duplicate_identifiers_rejected():
    with pytest.raises(InputError, match="duplicate"):
        parse_fasta(">a\nACD\n>a\nACD\n")


def test_empty_record_rejected():
    with pytest.raises(InputError):
        parse_fasta(">a\n\n>b\nACD\n")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("binder", BINDER), ("YES", BINDER), ("1", BINDER), ("positive", BINDER),
        ("non_binder", NON_BINDER), ("no", NON_BINDER), ("0", NON_BINDER),
        ("unknown", UNKNOWN), ("", UNKNOWN), ("n/a", UNKNOWN),
    ],
)
def test_binding_call_synonyms(raw, expected):
    assert normalise_call(raw) == expected


def test_unrecognised_call_rejected():
    with pytest.raises(InputError, match="not understood"):
        normalise_call("maybe")


def test_load_binding_csv(tmp_path):
    path = tmp_path / "binding.csv"
    path.write_text("species,binding\nmouse,binder\nhuman,non_binder\n")
    assert load_binding_calls(path) == {"mouse": BINDER, "human": NON_BINDER}


def test_load_binding_yaml(tmp_path):
    pytest.importorskip("yaml")
    path = tmp_path / "binding.yaml"
    path.write_text("binding:\n  mouse: binder\n  human: non_binder\n")
    assert load_binding_calls(path) == {"mouse": BINDER, "human": NON_BINDER}


def test_unknown_species_are_aligned_but_not_scored():
    calls = {"mouse": "binder", "rat": "binder", "human": "non_binder"}
    dataset, reference = build_dataset(records(), calls, "mouse")
    marmoset = dataset.get("marmoset")
    assert marmoset.call == UNKNOWN
    assert marmoset not in dataset.scored
    assert any("no binding call" in w for w in dataset.warnings)


def test_reference_must_be_a_binder():
    calls = {"mouse": "binder", "human": "non_binder"}
    with pytest.raises(InputError, match="must be a binder"):
        build_dataset(records(), calls, "human")


def test_reference_alias_resolution_is_case_insensitive():
    calls = {"MOUSE": "binder", "Human": "non_binder"}
    dataset, reference = build_dataset(records(), calls, "Mouse")
    assert reference == "mouse"
    assert dataset.get("mouse").call == BINDER


def test_binding_call_for_unknown_species_is_an_error():
    calls = {"mouse": "binder", "human": "non_binder", "zebrafish": "binder"}
    with pytest.raises(InputError, match="no matching sequence"):
        build_dataset(records(), calls, "mouse")


def test_both_groups_required():
    with pytest.raises(InputError, match="non_binder"):
        build_dataset(records(), {"mouse": "binder", "rat": "binder"}, "mouse")
