"""Residue equivalences from superposed structures."""

import pytest

from epitope_map import demo
from epitope_map.align import ResidueMap, align_sequences
from epitope_map.equivalence import (
    load_species_structures,
    structural_equivalence,
)
from epitope_map.io_seq import SpeciesRecord
from epitope_map.structure import load_structure


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    directory = tmp_path_factory.mktemp("equivalence")
    paths = demo.write_inputs(directory)
    reference = load_structure(str(paths["structure"]))
    seqs, _ = demo.species_sequences()
    records = [SpeciesRecord(name=n, sequence=s) for n, s in seqs.items()]
    alignment = align_sequences(records, reference="mouse")
    residue_map = ResidueMap.build(alignment, reference)
    return directory, residue_map, reference


def test_geometry_reproduces_a_correct_sequence_equivalence(context):
    """Where the alignment is right, an independent method must agree with it."""
    directory, residue_map, reference = context
    model = load_structure(
        str(demo.write_species_structure(directory / "human.pdb", "human"))
    )
    result = structural_equivalence(residue_map, reference, "human", model)
    assert result.usable
    assert result.n_superposed >= 20
    assert result.rmsd < 0.5
    assert not result.disagreements
    for ref_index, position in result.mapping.items():
        assert position == residue_map.species_index("human", ref_index)


def test_a_truncated_model_still_gives_full_sequence_positions(context):
    """The construct is not the sequence: an unmodelled N-terminus is an offset."""
    directory, residue_map, reference = context
    model = load_structure(
        str(
            demo.write_species_structure(
                directory / "human_trunc.pdb", "human", first=12
            )
        )
    )
    result = structural_equivalence(residue_map, reference, "human", model)
    assert result.usable
    # positions are reported in the species' own full-sequence numbering, not in
    # the index of the modelled fragment
    sample = {k: v for k, v in result.mapping.items() if k > 40}
    assert sample
    for ref_index, position in sample.items():
        assert position == residue_map.species_index("human", ref_index)


def test_renumbering_the_file_does_not_change_the_answer(context):
    """Author numbering is the file's business; mutants are named in sequence."""
    directory, residue_map, reference = context
    plain = load_structure(
        str(demo.write_species_structure(directory / "h1.pdb", "human"))
    )
    shifted = load_structure(
        str(demo.write_species_structure(directory / "h2.pdb", "human", shift=400))
    )
    a = structural_equivalence(residue_map, reference, "human", plain)
    b = structural_equivalence(residue_map, reference, "human", shifted)
    assert a.mapping == b.mapping


def test_a_structure_of_the_wrong_protein_is_refused(context, tmp_path):
    directory, residue_map, reference = context
    model = load_structure(
        str(demo.write_species_structure(directory / "h3.pdb", "human"))
    )
    result = structural_equivalence(residue_map, reference, "rat", model)
    assert not result.usable
    assert result.warnings
    assert "does not match its sequence" in result.warnings[0]


def test_too_few_anchors_falls_back_rather_than_guessing(context):
    directory, residue_map, reference = context
    model = load_structure(
        str(demo.write_species_structure(directory / "h4.pdb", "human", first=110))
    )
    result = structural_equivalence(residue_map, reference, "human", model)
    assert not result.usable
    assert any("too few to superpose" in w for w in result.warnings)


def test_species_structure_spec_parsing(context):
    directory, _, _ = context
    path = demo.write_species_structure(directory / "h5.pdb", "human")
    loaded = load_species_structures([f"human={path}"])
    assert set(loaded) == {"human"}
    with pytest.raises(ValueError, match="species=structure"):
        load_species_structures([str(path)])


def test_mutants_prefer_the_structural_position_when_it_differs():
    """The whole point: name the residue geometry picks, not the aligner's."""
    from epitope_map.patches import Patch
    from epitope_map.score import ResidueAnalysis
    from epitope_map.suggest import suggest_mutants

    class FakeMap:
        reference = "mouse"

        def species_index(self, species, ref_index):
            return 41  # what the alignment claims

    member = ResidueAnalysis(
        ref_index=40, column=40, aa="S", ref_number="65", rsa=0.6, discrimination=0.8
    )
    member.species_residues = {"mouse": "S", "human": "W"}
    patch = Patch(patch_id="PA", members=[member])

    class FakeDataset:
        non_binders = [SpeciesRecord(name="human", sequence="X", call="non_binder")]

    sequence_only = suggest_mutants([patch], FakeDataset(), FakeMap())
    gain = [m for m in sequence_only if m.direction == "gain_of_binding"][0]
    assert gain.position == 41

    structural = suggest_mutants(
        [patch], FakeDataset(), FakeMap(), equivalences={"human": {40: 38}}
    )
    gain = [m for m in structural if m.direction == "gain_of_binding"][0]
    assert gain.position == 38
    assert gain.label == "W38S"
