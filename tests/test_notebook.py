"""The one-line input parsing the Colab form boxes force on us."""

import pytest

from epitope_map.io_seq import InputError
from epitope_map.notebook import (
    check_panel,
    parse_binding_calls,
    sequence_labels,
    summarise,
    write_binding_csv,
)

EXPECTED = {"mouse": "binder", "rat": "binder", "human": "non_binder"}


@pytest.mark.parametrize(
    "text",
    [
        "mouse=binder, rat=binder, human=non_binder",
        "mouse=binder,rat=binder,human=non_binder",
        "mouse: binder; rat: binder; human: non_binder",
        "mouse,binder; rat,binder; human,non_binder",
        "mouse,binder,rat,binder,human,non_binder",
        "mouse,binder\nrat,binder\nhuman,non_binder",
        "mouse,binder\\nrat,binder\\nhuman,non_binder",  # literal backslash-n
        "species,binding\nmouse,binder\nrat,binder\nhuman,non_binder",
        "  mouse = binder ,  rat = binder ,  human = non_binder  ",
    ],
)
def test_every_reasonable_one_line_shape_is_accepted(text):
    assert parse_binding_calls(text) == EXPECTED


def test_call_synonyms_are_accepted():
    assert parse_binding_calls("mouse=yes, human=no, cyno=unknown") == {
        "mouse": "binder",
        "human": "non_binder",
        "cyno": "unknown",
    }


def test_species_labels_may_contain_spaces_and_dots():
    calls = parse_binding_calls("M. musculus=binder; rhesus macaque=non_binder")
    assert calls == {"M. musculus": "binder", "rhesus macaque": "non_binder"}


def test_unreadable_input_says_what_was_expected():
    with pytest.raises(InputError, match="mouse=binder"):
        parse_binding_calls("mouse")
    with pytest.raises(InputError, match="odd number"):
        parse_binding_calls("mouse,binder,rat")
    with pytest.raises(InputError, match="not understood"):
        parse_binding_calls("mouse=maybe")
    with pytest.raises(InputError, match="twice"):
        parse_binding_calls("mouse=binder, mouse=non_binder")
    with pytest.raises(InputError, match="no binding calls"):
        parse_binding_calls("   ")


def test_written_csv_is_what_the_cli_reads(tmp_path):
    from epitope_map.io_seq import load_binding_calls

    path = write_binding_csv(parse_binding_calls("mouse=binder, human=no"), tmp_path / "b.csv")
    assert load_binding_calls(path) == {"mouse": "binder", "human": "non_binder"}


def test_sequence_labels_from_accessions_and_fasta(tmp_path):
    assert sequence_labels("mouse=Q61503,rat=P21590,human=P21589") == [
        "mouse", "rat", "human",
    ]
    assert sequence_labels("Q61503, P21589") == ["Q61503", "P21589"]
    fasta = tmp_path / "s.fasta"
    fasta.write_text(">mouse desc\nACDE\n>human\nACDF\n")
    assert sequence_labels(str(fasta)) == ["mouse", "human"]


def test_panel_check_flags_a_binding_call_with_no_sequence():
    problems, _ = check_panel(EXPECTED, ["mouse", "rat"], "mouse")
    assert any("human" in p and "no sequence" in p for p in problems)


def test_panel_check_flags_a_missing_group():
    problems, _ = check_panel({"mouse": "binder", "rat": "binder"}, ["mouse", "rat"], "mouse")
    assert any("non_binder" in p for p in problems)


def test_panel_check_flags_a_non_binding_reference():
    problems, _ = check_panel(EXPECTED, ["mouse", "rat", "human"], "human")
    assert any("must be a species the antibody binds" in p for p in problems)
    problems, _ = check_panel(EXPECTED, ["mouse", "rat", "human"], "hamster")
    assert any("not in the binding table" in p for p in problems)


def test_panel_check_is_quiet_when_the_panel_is_fine():
    problems, notes = check_panel(EXPECTED, ["mouse", "rat", "human"], "mouse")
    assert problems == []
    assert not any("only" in note for note in notes)


def test_panel_check_notes_unscored_and_small_panels():
    calls = {"mouse": "binder", "human": "non_binder", "cyno": "unknown"}
    _, notes = check_panel(calls, ["mouse", "human", "cyno", "rat"], "mouse")
    assert any("rat" in note and "not used for scoring" in note for note in notes)
    assert any("only 2 species will be scored" in note for note in notes)


def test_summary_marks_the_reference():
    text = summarise(EXPECTED, "mouse")
    assert "<- reference" in text
    assert text.count("\n") == 4  # header, rule, three species
