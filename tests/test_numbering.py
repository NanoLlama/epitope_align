"""Numbering conversions - the spec calls this the main source of bugs."""

import pytest

from epitope_map import demo as synthetic
from epitope_map.align import AlignmentError, Alignment, ResidueMap, align_sequences
from epitope_map.io_seq import SpeciesRecord
from epitope_map.structure import load_structure


@pytest.fixture(scope="module")
def mapped(tmp_path_factory):
    directory = tmp_path_factory.mktemp("numbering")
    paths = synthetic.write_inputs(directory)
    structure = load_structure(str(paths["structure"]))
    seqs, _ = synthetic.species_sequences()
    records = [SpeciesRecord(name=n, sequence=s) for n, s in seqs.items()]
    alignment = align_sequences(records, reference="mouse")
    return ResidueMap.build(alignment, structure), structure, alignment


def test_author_numbering_offset_is_not_sequential_index(mapped):
    residue_map, _, _ = mapped
    # the synthetic structure starts at author number 25, sequence index 0
    assert residue_map.number_of(0) == str(synthetic.FIRST_RESSEQ)
    assert residue_map.ref_index_of_number(str(synthetic.FIRST_RESSEQ)) == 0


def test_insertion_code_round_trips(mapped):
    residue_map, structure, _ = mapped
    inserted = [r.key for r in structure.residues if r.key.icode.strip()]
    assert inserted, "fixture should contain an insertion code"
    key = inserted[0]
    ref_index = residue_map.ref_index_of_key(key)
    assert ref_index is not None
    assert residue_map.number_of(ref_index) == key.label
    assert key.label[-1].isalpha()
    assert residue_map.ref_index_of_number(key.label) == ref_index


def test_residues_missing_from_atom_records_stay_in_the_table(mapped):
    residue_map, _, _ = mapped
    unmodelled = [
        position for position in residue_map.positions() if not position.modelled
    ]
    assert len(unmodelled) == len(synthetic.UNMODELLED_INDICES)
    for position in unmodelled:
        assert position.number is None
        assert position.aa == residue_map.ref_seq[position.ref_index]


def test_column_and_reference_index_are_inverse(mapped):
    residue_map, _, _ = mapped
    for position in residue_map.positions():
        assert residue_map.ref_index_of_column(position.column) == position.ref_index
        assert residue_map.column_of(position.ref_index) == position.column


def test_species_index_is_in_that_species_own_numbering(mapped):
    residue_map, _, alignment = mapped
    # macaque carries a deletion, so its own numbering runs behind the reference
    last = len(residue_map) - 1
    mouse_position = residue_map.species_index("mouse", last)
    macaque_position = residue_map.species_index("macaque", last)
    assert mouse_position == last + 1
    assert macaque_position is not None
    assert macaque_position < mouse_position


def test_gap_columns_have_no_species_index(mapped):
    residue_map, _, alignment = mapped
    gapped = [
        position.ref_index
        for position in residue_map.positions()
        if alignment.sequences["macaque"][position.column] == "-"
    ]
    assert gapped
    assert residue_map.species_index("macaque", gapped[0]) is None


def test_ectodomain_in_structure_and_sequence_numbering(mapped):
    residue_map, structure, alignment = mapped
    by_structure = ResidueMap.build(
        alignment, structure, ectodomain=(30, 40), ectodomain_numbering="structure"
    )
    by_sequence = ResidueMap.build(
        alignment, structure, ectodomain=(30, 40), ectodomain_numbering="sequence"
    )
    structure_positions = [
        p.number for p in by_structure.positions() if p.in_ectodomain
    ]
    sequence_positions = [
        p.ref_index + 1 for p in by_sequence.positions() if p.in_ectodomain
    ]
    assert structure_positions == [str(n) for n in range(30, 41)]
    assert sequence_positions == list(range(30, 41))
    # the same range means different residues under the two conventions
    assert by_structure.ref_index_of_number("30") != 29


def test_mismatched_structure_fails_loudly(mapped):
    residue_map, structure, alignment = mapped
    scrambled = dict(alignment.sequences)
    reference = list(scrambled["mouse"])
    for i in range(0, len(reference), 3):
        reference[i] = "W" if reference[i] != "W" else "G"
    scrambled["mouse"] = "".join(reference)
    broken = Alignment(sequences=scrambled, reference="mouse", method="test")
    with pytest.raises(AlignmentError) as excinfo:
        ResidueMap.build(broken, structure)
    message = str(excinfo.value)
    assert "disagree" in message
    assert "structure" in message  # the diff is shown, not swallowed
