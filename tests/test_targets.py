"""Built-in target profiles."""

import pytest

from epitope_map.cli import build_parser, config_from_args, main
from epitope_map.io_seq import InputError
from epitope_map.targets import available_targets, describe_targets, load_profile


def test_tfr1_profile_contents():
    profile = load_profile("tfr1")
    assert profile.reference == "mouse"
    assert profile.sequences["mouse"] == "Q62351"
    assert profile.sequences["human"] == "P02786"
    assert len(profile.sequences) == 4

    # the domains UniProt does not have, which is the point of the profile
    names = [d["name"] for d in profile.domains]
    assert names == ["protease-like-N", "apical", "protease-like-C", "helical"]
    apical = next(d for d in profile.domains if d["name"] == "apical")
    assert (apical["start"], apical["end"]) == (189, 383)

    assert profile.topology_value() == (
        "cytoplasmic=1-67,transmembrane=68-88,extracellular=89-763"
    )
    assert profile.oligomer == "homodimer"
    assert profile.interface_ranges() == [(607, 760)]
    assert len(profile.candidate_species) == 9
    assert profile.candidate_species["guinea_pig"] == "G4WF79"
    assert {a.species for a in profile.avoid} == {"cynomolgus", "cow"}
    assert any("TFR2" in note for note in profile.notes)
    assert any("carboxypeptidase" in note for note in profile.notes)


def test_profile_renders_cli_shaped_values():
    profile = load_profile("tfr1")
    assert profile.sequences_value().startswith("mouse=Q62351,")
    assert "guinea_pig=G4WF79" in profile.candidates_value()
    table = profile.domains_table().splitlines()
    assert table[0] == "name\tstart\tend"
    assert "apical\t189\t383" in table


def test_listing_targets():
    assert "tfr1" in available_targets()
    text = describe_targets()
    assert "tfr1" in text and "TfR1" in text
    assert "--target" in text


def test_unknown_target_lists_what_exists():
    with pytest.raises(InputError) as excinfo:
        load_profile("no_such_target")
    assert "tfr1" in str(excinfo.value)


def test_list_targets_exits_cleanly(capsys):
    assert main(["--list-targets"]) == 0
    assert "tfr1" in capsys.readouterr().out


def test_profile_fills_defaults_and_records_where_they_came_from(tmp_path):
    args = build_parser().parse_args(
        [
            "--target", "tfr1",
            "--binding", "binding.csv",
            "--structure", "AF-Q62351-F1",
            "--outdir", str(tmp_path),
        ]
    )
    config = config_from_args(args)
    assert config.sequences.startswith("mouse=Q62351")
    assert config.reference == "mouse"
    assert "extracellular=89-763" in config.topology
    assert len(config.candidate_species) == 9
    assert config.domains  # a table was written out for the run

    assert config.provenance["sequences"] == "profile:tfr1"
    assert config.provenance["reference"] == "profile:tfr1"
    assert config.provenance["topology"] == "profile:tfr1"
    assert config.provenance["domains"] == "profile:tfr1"
    # the structure is deliberately not chosen for the user
    assert "structure" not in config.provenance
    assert config.structure == "AF-Q62351-F1"


def test_explicit_flags_beat_the_profile(tmp_path):
    args = build_parser().parse_args(
        [
            "--target", "tfr1",
            "--binding", "binding.csv",
            "--structure", "AF-Q62351-F1",
            "--reference", "rat",
            "--topology", "whole-chain",
            "--candidate-species", "dog=Q9GLD3",
            "--outdir", str(tmp_path),
        ]
    )
    config = config_from_args(args)
    assert config.reference == "rat"
    assert config.topology == "whole-chain"
    assert config.candidate_species == ["dog=Q9GLD3"]
    for key in ("reference", "topology", "candidate_species"):
        assert key not in config.provenance  # the user's, not the profile's


def test_profile_domain_table_is_written_and_loadable(tmp_path):
    args = build_parser().parse_args(
        ["--target", "tfr1", "--binding", "b.csv", "--structure", "x",
         "--outdir", str(tmp_path)]
    )
    config = config_from_args(args)

    from epitope_map.topology import load_domains_tsv

    segments = load_domains_tsv(config.domains)
    assert [s.description for s in segments] == [
        "protease-like-N", "apical", "protease-like-C", "helical"
    ]


def test_known_bad_accession_is_refused(tmp_path, monkeypatch):
    """The cynomolgus entry with a 28-residue gene-model gap."""
    from epitope_map.io_seq import SpeciesRecord, build_dataset
    from epitope_map.pipeline import RunConfig, _check_against_profile

    profile = load_profile("tfr1")
    records = [
        SpeciesRecord(name="mouse", sequence="ACDE", accession="Q62351"),
        SpeciesRecord(name="cyno", sequence="ACDF", accession="A0A2K5X958"),
    ]
    dataset, reference = build_dataset(
        records, {"mouse": "binder", "cyno": "non_binder"}, "mouse"
    )
    config = RunConfig(
        sequences="", binding="", reference="mouse", structure="", profile=profile
    )

    class FakeAlignment:
        identities = {"cyno": 96.0}

    with pytest.raises(InputError) as excinfo:
        _check_against_profile(config, dataset, FakeAlignment(), "mouse")
    message = str(excinfo.value)
    assert "A0A2K5X958" in message
    assert "gene-model gap" in message


def test_a_paralog_level_identity_is_flagged():
    from epitope_map.io_seq import SpeciesRecord, build_dataset
    from epitope_map.pipeline import RunConfig, _check_against_profile

    profile = load_profile("tfr1")
    records = [
        SpeciesRecord(name="mouse", sequence="ACDE", accession="Q62351"),
        SpeciesRecord(name="mystery", sequence="ACDF", accession="Q9UP52"),
    ]
    dataset, _ = build_dataset(
        records, {"mouse": "binder", "mystery": "non_binder"}, "mouse"
    )
    config = RunConfig(
        sequences="", binding="", reference="mouse", structure="", profile=profile
    )

    class FakeAlignment:
        identities = {"mystery": 58.0}   # TFR2 territory

    notes = _check_against_profile(config, dataset, FakeAlignment(), "mouse")
    assert any("58.0% identical" in note and "paralog" in note for note in notes)


def test_target_profile_drives_a_whole_run(synthetic_inputs, tmp_path):
    """End to end: a profile supplies topology, domains and candidates."""
    import yaml

    from epitope_map import demo
    from epitope_map.cli import apply_profile
    from epitope_map.pipeline import RunConfig, run_pipeline

    # a profile shaped like tfr1 but sized for the synthetic protein
    seqs, _ = demo.species_sequences(outgroup=True)
    candidates = tmp_path / "candidates.fasta"
    candidates.write_text(f">guinea_pig_like\n{seqs['outgroup']}\n")

    directory = tmp_path / "targets"
    directory.mkdir()
    (directory / "toy.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "Toy receptor",
                "reference_default": "mouse",
                "domains": [
                    {"name": "stalk", "start": 1, "end": 40},
                    {"name": "apical", "start": 41, "end": 90},
                    {"name": "helical", "start": 91, "end": 120},
                ],
                "topology": {"cytoplasmic": [1, 20], "extracellular": [21, 120]},
                "oligomer": "homodimer",
                "oligomer_interface_domains": ["helical"],
                "candidate_species": {},
            }
        )
    )
    profile = load_profile("toy", directory)

    values = {
        "binding": str(synthetic_inputs["binding"]),
        "structure": str(synthetic_inputs["structure"]),
        "sequences": str(synthetic_inputs["sequences"]),
    }
    provenance = apply_profile(values, profile, tmp_path)
    assert provenance["topology"] == "profile:toy"
    assert provenance["domains"] == "profile:toy"

    result = run_pipeline(
        RunConfig(
            sequences=values["sequences"],
            binding=values["binding"],
            reference=str(values["reference"]),
            structure=values["structure"],
            topology=str(values["topology"]),
            domains=str(values["domains"]),
            candidate_species=[str(candidates)],
            outdir=tmp_path / "out",
            profile=profile,
            provenance=provenance,
        )
    )

    # the profile's domains are used, not a whole-chain contact-graph blob
    assert {d.name for d in result.structural_domains} == {"stalk", "apical", "helical"}
    assert result.domain_source.startswith("user table")
    # its topology is applied
    assert any(r.topology == "cytoplasmic" for r in result.residues)
    assert all(not r.accessible for r in result.residues if r.topology == "cytoplasmic")
    # its candidates are scored and named
    assert any(e["candidate"] == "guinea_pig_like" for e in result.panel_advice)
    assert result.panel_advice[0]["kind"] == "real ortholog"

    from epitope_map.report import write_all

    text = write_all(result, result.config.outdir)["report"].read_text()
    assert "Target profile: **Toy receptor**" in text
    assert "profile:toy" in text        # provenance is visible per parameter
    assert "| you |" in text            # and so is what the user chose


def test_a_patch_inside_one_profile_domain_gets_that_swap(synthetic_inputs, tmp_path):
    """The point of the profile: an apical-domain swap instead of 'whole chain'."""
    from epitope_map.pipeline import RunConfig, run_pipeline

    table = tmp_path / "domains.tsv"
    table.write_text("name\tstart\tend\nstalk\t1\t40\napical\t41\t90\nhelical\t91\t120\n")
    result = run_pipeline(
        RunConfig(
            sequences=str(synthetic_inputs["sequences"]),
            binding=str(synthetic_inputs["binding"]),
            reference="mouse",
            structure=str(synthetic_inputs["structure"]),
            topology="whole-chain",
            domains=str(table),
            outdir=tmp_path / "out",
        )
    )
    swaps = {c.patch_id: c.domain_swap for c in result.chimeras}
    assert swaps, "there should be chimera suggestions"
    # any swap that is offered names a real domain, never a whole-chain blob
    for swap in swaps.values():
        if swap:
            assert swap in ("dom:stalk", "dom:apical", "dom:helical")
