"""CLI argument handling, config-file mode and the fallback aligner path."""

import pytest

from epitope_map.cli import build_parser, config_from_args, main, parse_range
from epitope_map.io_seq import InputError


def test_parse_range():
    assert parse_range("25-240") == (25, 240)
    assert parse_range(" 25 - 240 ") == (25, 240)
    with pytest.raises(Exception):
        parse_range("240-25")
    with pytest.raises(Exception):
        parse_range("25")


def args_for(inputs, tmp_path, extra=()):
    argv = [
        "--sequences", str(inputs["sequences"]),
        "--binding", str(inputs["binding"]),
        "--reference", "mouse",
        "--structure", str(inputs["structure"]),
        "--outdir", str(tmp_path / "out"),
        *extra,
    ]
    return build_parser().parse_args(argv)


def test_config_from_command_line(synthetic_inputs, tmp_path):
    config = config_from_args(args_for(synthetic_inputs, tmp_path))
    assert config.reference == "mouse"
    assert config.rsa_cutoff == 0.20
    assert config.patch_radius == 12.0


def test_missing_required_options_are_reported():
    args = build_parser().parse_args([])
    with pytest.raises(InputError, match="missing required option"):
        config_from_args(args)


def test_config_file_mode(synthetic_inputs, tmp_path):
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sequences": str(synthetic_inputs["sequences"]),
                "binding": str(synthetic_inputs["binding"]),
                "reference": "mouse",
                "structure": str(synthetic_inputs["structure"]),
                "outdir": str(tmp_path / "out"),
                "ectodomain": "30-140",
                "rsa-cutoff": 0.3,
                "patch-radius": 10.0,
            }
        )
    )
    args = build_parser().parse_args(["--config", str(config_path)])
    config = config_from_args(args)
    assert config.reference == "mouse"
    assert config.ectodomain == (30, 140)
    assert config.rsa_cutoff == 0.3
    assert config.patch_radius == 10.0


def test_command_line_overrides_the_config_file(synthetic_inputs, tmp_path):
    yaml = pytest.importorskip("yaml")
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sequences": str(synthetic_inputs["sequences"]),
                "binding": str(synthetic_inputs["binding"]),
                "reference": "mouse",
                "structure": str(synthetic_inputs["structure"]),
                "rsa_cutoff": 0.3,
            }
        )
    )
    args = build_parser().parse_args(
        ["--config", str(config_path), "--rsa-cutoff", "0.45"]
    )
    assert config_from_args(args).rsa_cutoff == 0.45


def test_main_writes_every_output(synthetic_inputs, tmp_path, capsys):
    outdir = tmp_path / "out"
    code = main(
        [
            "--sequences", str(synthetic_inputs["sequences"]),
            "--binding", str(synthetic_inputs["binding"]),
            "--reference", "mouse",
            "--structure", str(synthetic_inputs["structure"]),
            "--topology", "whole-chain",
            "--outdir", str(outdir),
        ]
    )
    assert code == 0
    for name in (
        "residues.tsv", "patches.tsv", "chimeras.tsv", "mutants.tsv",
        "alignment.fasta", "session.pml", "report.md",
    ):
        assert (outdir / name).exists()
    captured = capsys.readouterr().out
    assert "patch(es)" in captured
    assert "WARNING" in captured  # the two-clade warning reaches the terminal


def test_main_exits_with_a_message_on_bad_input(synthetic_inputs, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--sequences", str(synthetic_inputs["sequences"]),
                "--binding", str(synthetic_inputs["binding"]),
                "--reference", "human",  # a non-binder as reference
                "--structure", str(synthetic_inputs["structure"]),
                "--outdir", str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    assert "must be a binder" in capsys.readouterr().err


def test_pairwise_fallback_warns_loudly(synthetic_inputs, tmp_path):
    from epitope_map.pipeline import RunConfig, run_pipeline

    config = RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(synthetic_inputs["binding"]),
        reference="mouse",
        structure=str(synthetic_inputs["structure"]),
        topology="whole-chain",
        outdir=tmp_path / "out",
        aligner="pairwise",
    )
    result = run_pipeline(config)
    assert result.alignment.method == "biopython-pairwise-to-reference"
    assert any("NO TRUE MSA" in w for w in result.warnings)


def test_demo_flag_runs_end_to_end(tmp_path, capsys):
    """The one-command check a non-coder is told to run first."""
    from epitope_map import demo

    assert main(["--demo", "--outdir", str(tmp_path)]) == 0
    results = tmp_path / "results"
    for name in ("report.md", "patches.tsv", "residues.tsv", "session.pml"):
        assert (results / name).exists()
    output = capsys.readouterr().out
    assert "Installation looks healthy" in output
    # the banner promises these residues; the run must actually produce them
    for number in demo.truth_numbers():
        assert number in output


def test_cli_stops_with_advice_when_topology_is_unknown(synthetic_inputs, tmp_path, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(
            [
                "--sequences", str(synthetic_inputs["sequences"]),
                "--binding", str(synthetic_inputs["binding"]),
                "--reference", "mouse",
                "--structure", str(synthetic_inputs["structure"]),
                "--outdir", str(tmp_path / "out"),
            ]
        )
    assert excinfo.value.code == 2
    message = capsys.readouterr().err
    assert "--topology" in message and "whole-chain" in message


def test_topology_flag_is_parsed_into_the_config(synthetic_inputs, tmp_path):
    args = args_for(
        synthetic_inputs, tmp_path,
        extra=["--topology", "extracellular=90-763,tm=68-88", "--radius-sweep", "10,14,18"],
    )
    config = config_from_args(args)
    assert config.topology == "extracellular=90-763,tm=68-88"
    assert config.radius_sweep == [10.0, 14.0, 18.0]


def test_sequences_may_be_written_as_a_mapping(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "sequences": {"mouse": "Q62351", "human": "P02786"},
                "binding": "b.csv",
                "reference": "mouse",
                "structure": "AF-Q62351-F1",
                "topology": "cytoplasmic=1-67,tm=68-88,extracellular=89-763",
                "disordered_region_exclusion": True,
                "assembly": "biological",
                "radius_sweep": "10,12,14",
            }
        )
    )
    config = config_from_args(build_parser().parse_args(["--config", str(path)]))
    # YAML mappings do not preserve order, and nothing downstream depends on it
    assert set(config.sequences.split(",")) == {"mouse=Q62351", "human=P02786"}
    assert config.topology.startswith("cytoplasmic=1-67")
    assert config.keep_disordered is False
    assert config.prefer_assembly is True
    assert config.radius_sweep == [10.0, 12.0, 14.0]


def test_shipped_tfr1_config_parses():
    """The worked configuration for the target the change requests came from."""
    config = config_from_args(
        build_parser().parse_args(["--config", "examples/tfr1.yaml"])
    )
    assert "mouse=Q62351" in config.sequences
    assert config.reference == "mouse"
    assert "extracellular=89-763" in config.topology
    assert config.radius_sweep == [10.0, 12.0, 14.0, 16.0, 18.0]
