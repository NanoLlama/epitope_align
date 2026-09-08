"""Alignment reliability: the failure mode that costs real money.

A patch can sit in the right region while the specific residue equivalences
inside it are the aligner's guess. Ordering a point mutant on that guess wastes
a construct, so the pipeline has to know the difference.
"""

import pytest

from epitope_map.align import (
    CONFIDENCE_CUTOFF,
    align_sequences,
    alignment_confidence,
    local_identity,
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


def test_confidence_is_lower_in_the_ambiguous_window(reliability):
    """Regression test 4: the indel-rich loop must not look as safe as the frame."""
    _, confidence, _ = reliability
    head = range(len(CONSERVED_HEAD))
    window = range(len(CONSERVED_HEAD), len(CONSERVED_HEAD) + len(AMBIGUOUS["mouse"]))

    frame_mean = sum(confidence[i] for i in head) / len(head)
    window_mean = sum(confidence[i] for i in window) / len(window)
    assert frame_mean > window_mean
    assert frame_mean > 0.9
    assert min(confidence[i] for i in window) < CONFIDENCE_CUTOFF


def test_confidence_reports_how_it_was_computed(reliability):
    _, _, notes = reliability
    assert notes
    assert any("alignment confidence" in note for note in notes)


def test_local_identity_dips_in_the_ambiguous_window(reliability):
    alignment, _, _ = reliability
    identity = local_identity(alignment, window=8)
    frame = identity[5]
    middle = identity[len(CONSERVED_HEAD) + len(AMBIGUOUS["mouse"]) // 2]
    assert frame > middle


def test_conserved_alignment_is_confident_throughout():
    records = [
        SpeciesRecord(name=name, sequence=CONSERVED_HEAD + CONSERVED_TAIL)
        for name in ("mouse", "rat", "human")
    ]
    records[2].sequence = records[2].sequence.replace("GYTLDD", "GYTLED")
    alignment = align_sequences(records, reference="mouse")
    confidence, _ = alignment_confidence(alignment, records)
    assert min(confidence.values()) >= CONFIDENCE_CUTOFF


def test_unverified_mutants_are_marked_and_deprioritised():
    from epitope_map.score import ResidueAnalysis
    from epitope_map.suggest import _reliability

    solid = ResidueAnalysis(ref_index=1, column=1, aa="K", alignment_confidence=1.0)
    shaky = ResidueAnalysis(ref_index=2, column=2, aa="K", alignment_confidence=0.4)
    loose = ResidueAnalysis(ref_index=3, column=3, aa="K", alignment_confidence=1.0)
    loose.low_identity_window = True
    loose.local_identity = 41.0

    assert _reliability(solid) == (False, "")
    assert _reliability(shaky)[0] is True
    assert "alignment confidence 0.40" in _reliability(shaky)[1]
    assert "swap the segment" in _reliability(shaky)[1]
    assert _reliability(loose)[0] is True
    assert "local identity 41%" in _reliability(loose)[1]
