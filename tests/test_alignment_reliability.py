"""Alignment reliability: the failure mode that costs real money.

A patch can sit in the right region while the specific residue equivalences
inside it are the aligner's guess. Ordering a point mutant on that guess wastes
a construct, so the pipeline has to know the difference.
"""

import pytest

from epitope_map.align import (
    CONFIDENCE_CUTOFF,
    MIN_DECORRELATED_METHODS,
    align_sequences,
    alignment_confidence,
    local_identity,
    low_identity_windows,
)
from epitope_map.io_seq import SpeciesRecord

# a conserved frame around a low-identity, indel-bearing window - the shape of
# the real case, where the mouse->human offset shifted inside the patch
CONSERVED_HEAD = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
CONSERVED_TAIL = "GYTLDDLQKAVEFLDRQTGCDVEIVSTKRVLDA"
AMBIGUOUS = {
    "mouse": "QVKSSIGQNMVTIVQSNGNLDPVESPEGY",
    "rat": "QVKNSVSQNLVTINSGSNIDPVEAPEGY",
    "human": "QVKDSAQNSVIIVDKNGRLVYLVENPGGY",
    "marmoset": "QVKDSAQNSVTITGTNSEFVYLVENPGGY",
}


def _records():
    return [
        SpeciesRecord(name=name, sequence=CONSERVED_HEAD + middle + CONSERVED_TAIL)
        for name, middle in AMBIGUOUS.items()
    ]


@pytest.fixture(scope="module")
def reliability():
    records = _records()
    alignment = align_sequences(records, reference="mouse")
    confidence, notes = alignment_confidence(alignment, records)
    return alignment, confidence, notes


def _decorrelated_methods_available():
    import shutil

    return sum(
        1 for tool in ("muscle", "clustalo", "t_coffee") if shutil.which(tool)
    ) + (1 if shutil.which("mafft") else 0)


def test_confidence_is_blank_rather_than_1_when_nothing_could_disagree(reliability):
    """Regression test 4: agreement between settings of one program is not evidence."""
    if _decorrelated_methods_available() >= MIN_DECORRELATED_METHODS:
        pytest.skip("enough independent aligners are installed to compute it")
    _, confidence, notes = reliability
    assert confidence == {}                      # not a page of 1.0s
    assert any("UNINFORMATIVE" in note for note in notes)
    assert any("not evidence" in note for note in notes)
    assert any("Install MUSCLE" in note for note in notes)


def test_confidence_reports_how_it_was_computed(reliability):
    _, _, notes = reliability
    assert notes
    assert any(
        "alignment confidence" in note or "alignment_confidence" in note
        for note in notes
    )


def test_local_identity_dips_in_the_ambiguous_window(reliability):
    alignment, _, _ = reliability
    identity = local_identity(alignment, window=8)
    frame = identity[5]
    middle = identity[len(CONSERVED_HEAD) + len(AMBIGUOUS["mouse"]) // 2]
    assert frame > middle


def test_ambiguity_is_flagged_as_a_window_not_residue_by_residue(reliability):
    """The whole loop is uncertain; a cutoff crossing inside it means nothing."""
    alignment, _, _ = reliability
    identity = local_identity(alignment, window=8)
    windows = low_identity_windows(identity)
    assert windows, "the indel-rich loop should be detected"

    ambiguous = range(
        len(CONSERVED_HEAD), len(CONSERVED_HEAD) + len(AMBIGUOUS["mouse"])
    )
    covered = {i for w in windows for i in range(w.start, w.end + 1)}
    overlap = covered & set(ambiguous)
    assert len(overlap) >= 0.6 * len(ambiguous)

    # and the conserved frame is left alone
    assert not covered & set(range(10))

    window = windows[0]
    assert 0 <= window.identity <= 100
    assert "identity" in window.label()


def test_a_uniformly_conserved_alignment_has_no_ambiguous_window():
    records = [
        SpeciesRecord(name=name, sequence=CONSERVED_HEAD + CONSERVED_TAIL)
        for name in ("mouse", "rat", "human")
    ]
    records[2].sequence = records[2].sequence.replace("GYTLDD", "GYTLED")
    alignment = align_sequences(records, reference="mouse")
    assert low_identity_windows(local_identity(alignment, window=8)) == []


def test_window_marks_every_residue_in_it_uniformly():
    """The reported failure: 61% flagged, 67% and 63% not, all one loop."""
    identity = {i: 90.0 for i in range(60)}
    for i, value in zip(range(20, 32), [67.5, 61.0, 63.4, 58.0, 62.0, 66.0,
                                        59.0, 64.0, 61.5, 68.0, 62.5, 65.0]):
        identity[i] = value
    windows = low_identity_windows(identity)
    assert len(windows) == 1
    window = windows[0]
    assert window.start <= 20 and window.end >= 31
    assert all(window.contains(i) for i in range(20, 32))


def test_unverified_mutants_are_marked_and_deprioritised():
    from epitope_map.score import ResidueAnalysis
    from epitope_map.suggest import _reliability

    solid = ResidueAnalysis(ref_index=1, column=1, aa="K", alignment_confidence=1.0)
    shaky = ResidueAnalysis(ref_index=2, column=2, aa="K", alignment_confidence=0.4)
    loose = ResidueAnalysis(ref_index=3, column=3, aa="K", alignment_confidence=1.0)
    loose.low_identity_window = True
    loose.local_identity = 41.0
    loose.identity_window = "195-235 (58% identity)"

    assert _reliability(solid) == (False, "")
    assert _reliability(shaky)[0] is True
    assert "alignment confidence 0.40" in _reliability(shaky)[1]
    assert "swap the segment" in _reliability(shaky)[1]
    assert _reliability(loose)[0] is True
    # the window is named, not the individual residue's identity value
    assert "195-235 (58% identity)" in _reliability(loose)[1]


def test_patch_containing_ambiguous_columns_is_flagged():
    """A patch is a region-level claim; its residue pairings may still be guesses."""
    from epitope_map.patches import find_patches
    from epitope_map.score import ResidueAnalysis

    def residue(index, confidence):
        row = ResidueAnalysis(
            ref_index=index,
            column=index,
            aa="K",
            ref_number=str(index + 1),
            discrimination=0.8,
            modelled=True,
            centroid=(index * 3.0, 0.0, 0.0),
            rsa=0.5,
            alignment_confidence=confidence,
        )
        row.composite = 0.8
        return row

    clean, _ = find_patches(
        [residue(i, 1.0) for i in range(3)], discrimination_cutoff=0.25, radius=12.0
    )
    assert not any("ambiguous" in flag for flag in clean[0].flags)

    shaky, _ = find_patches(
        [residue(0, 1.0), residue(1, 0.4), residue(2, 0.5)],
        discrimination_cutoff=0.25,
        radius=12.0,
    )
    flag = next(f for f in shaky[0].flags if "ambiguous" in f)
    assert "2 of 3" in flag
    assert "swap the segment" in flag
