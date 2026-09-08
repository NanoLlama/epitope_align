"""Every documented shape of --candidate-species must load.

Three runs shipped with the candidates silently missing because a
comma-separated list arrived from argparse as a single unsplit token and was
rejected as an unrecognised accession.
"""

import pytest

from epitope_map.cli import build_parser, config_from_args
from epitope_map.io_seq import InputError, load_sequences

FASTA = ">syrian_hamster A0A1U7QRP4\nACDEFGHIKLMNPQRSTVWY\n>dog Q9GLD3\nACDEFGHIKLMNPQRSTVWA\n"


class _Response:
    status_code = 200

    def __init__(self, accession):
        self.text = f">sp|{accession}|TFR1_TEST Transferrin receptor protein 1\nACDEFGHIKLMNPQRSTVWY\n"

    def json(self):
        return {}


@pytest.fixture
def offline_uniprot(monkeypatch):
    """Answer UniProt locally, so the accession paths are testable offline."""
    seen = []

    def fake_get(url, timeout=60):
        accession = url.rsplit("/", 1)[-1].split(".")[0]
        seen.append(accession)
        return _Response(accession)

    monkeypatch.setattr("epitope_map.io_seq._uniprot_get", fake_get)
    return seen


def _candidates(value):
    args = build_parser().parse_args(
        [
            "--sequences", "a.fasta", "--binding", "b.csv",
            "--reference", "mouse", "--structure", "x.pdb",
            "--candidate-species", value,
        ]
    )
    return config_from_args(args).candidate_species


def test_label_accession_comma_list(offline_uniprot, tmp_path):
    """The format the report itself documents - and the one that was rejected."""
    value = "syrian_hamster=A0A1U7QRP4, chinese_hamster=Q07891, cat=Q9MYZ3"
    items = _candidates(value)
    assert items == [
        "syrian_hamster=A0A1U7QRP4",
        "chinese_hamster=Q07891",
        "cat=Q9MYZ3",
    ]
    records = load_sequences(items, cache_dir=tmp_path)
    assert [r.name for r in records] == ["syrian_hamster", "chinese_hamster", "cat"]
    # each accession is fetched twice: the sequence, then the feature table
    assert sorted(set(offline_uniprot)) == ["A0A1U7QRP4", "Q07891", "Q9MYZ3"]
    assert len(offline_uniprot) == 6


def test_bare_accession_comma_list(offline_uniprot, tmp_path):
    items = _candidates("Q07891,Q9MYZ3")
    assert items == ["Q07891", "Q9MYZ3"]
    records = load_sequences(items, cache_dir=tmp_path)
    assert len(records) == 2


def test_fasta_path(tmp_path):
    path = tmp_path / "candidates.fasta"
    path.write_text(FASTA)
    items = _candidates(str(path))
    assert items == [str(path)]
    records = load_sequences(items)
    # the label is the header up to the first whitespace
    assert [r.name for r in records] == ["syrian_hamster", "dog"]


def test_repeated_flag_and_mixtures(offline_uniprot, tmp_path):
    path = tmp_path / "candidates.fasta"
    path.write_text(FASTA)
    args = build_parser().parse_args(
        [
            "--sequences", "a.fasta", "--binding", "b.csv",
            "--reference", "mouse", "--structure", "x.pdb",
            "--candidate-species", str(path),
            "--candidate-species", "cat=Q9MYZ3,dog2=Q9GLD3",
        ]
    )
    items = config_from_args(args).candidate_species
    assert items == [str(path), "cat=Q9MYZ3", "dog2=Q9GLD3"]
    records = load_sequences(items, cache_dir=tmp_path)
    assert [r.name for r in records] == ["syrian_hamster", "dog", "cat", "dog2"]


def test_missing_file_says_so_and_looks_for_it(tmp_path, monkeypatch):
    real = tmp_path / "elsewhere" / "tfr1_candidates.fasta"
    real.parent.mkdir()
    real.write_text(FASTA)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(InputError) as excinfo:
        load_sequences(["/content/my-run/tfr1_candidates.fasta"])
    message = str(excinfo.value)
    assert "no such file" in message
    assert "does not exist either" in message
    # and it points at the copy that does exist
    assert "tfr1_candidates.fasta" in message
    assert "elsewhere" in message


def test_unreadable_fasta_names_the_parser_error(tmp_path):
    path = tmp_path / "broken.fasta"
    path.write_text("this is not fasta at all\n")
    with pytest.raises(InputError, match="not readable as FASTA"):
        load_sequences([str(path)])


def test_unrecognised_token_lists_the_accepted_formats():
    with pytest.raises(InputError) as excinfo:
        load_sequences(["not_an_accession"])
    message = str(excinfo.value)
    assert "Accepted formats" in message
    assert "label=ACCESSION" in message
    assert "FASTA file path" in message


def test_fetch_failure_names_the_accession(monkeypatch, tmp_path):
    class Missing:
        status_code = 404
        text = ""

    monkeypatch.setattr("epitope_map.io_seq._uniprot_get", lambda url, timeout=60: Missing())
    with pytest.raises(InputError) as excinfo:
        load_sequences(["Q07891"], cache_dir=tmp_path)
    assert "Q07891" in str(excinfo.value)
    assert "404" in str(excinfo.value)
    assert "no such accession" in str(excinfo.value)


def test_network_failure_is_not_a_traceback(monkeypatch, tmp_path):
    def explode(url, timeout=60):
        raise OSError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr("epitope_map.io_seq._uniprot_get", explode)
    with pytest.raises(OSError):
        load_sequences(["Q07891"], cache_dir=tmp_path)


def test_unloadable_candidates_stop_the_run_loudly(synthetic_inputs, tmp_path, capsys):
    """An ignored flag must not be one warning among eleven."""
    from epitope_map.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--sequences", str(synthetic_inputs["sequences"]),
                "--binding", str(synthetic_inputs["binding"]),
                "--reference", "mouse",
                "--structure", str(synthetic_inputs["structure"]),
                "--topology", "whole-chain",
                "--candidate-species", "/no/such/file.fasta",
                "--outdir", str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert "INPUT IGNORED - NOTHING WAS RUN" in message
    assert "no such file" in message
    assert not (tmp_path / "out" / "report.md").exists()
