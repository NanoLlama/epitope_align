"""Discrimination scoring, gap/missing handling, degeneracy calibration."""

import pytest

from epitope_map.io_seq import SpeciesRecord, build_dataset
from epitope_map.score import (
    GAP_DIFFERENCE,
    MISSING,
    assess_degeneracy,
    composite_score,
    confidence_weight,
    exposure_weight,
    pair_difference,
    pattern_consistency,
    score_column,
    upgma_root_split,
)

BINDERS = ["mouse", "rat"]
NON_BINDERS = ["human", "marmoset"]


def discrimination(residues):
    return score_column(residues, BINDERS, NON_BINDERS)[0]


def test_clean_discriminating_position_scores_high():
    clean = discrimination({"mouse": "K", "rat": "K", "human": "W", "marmoset": "W"})
    noisy = discrimination({"mouse": "K", "rat": "W", "human": "W", "marmoset": "K"})
    assert clean > 0.5
    assert noisy < 0


def test_variation_within_binders_is_penalised():
    conserved = discrimination({"mouse": "K", "rat": "K", "human": "D", "marmoset": "D"})
    varying = discrimination({"mouse": "K", "rat": "Y", "human": "D", "marmoset": "D"})
    assert conserved > varying


def test_identical_position_scores_zero():
    assert discrimination({n: "A" for n in BINDERS + NON_BINDERS}) == 0.0


def test_gap_versus_residue_is_a_strong_signal():
    value = pair_difference("-", "K")
    assert value == GAP_DIFFERENCE
    assert value > pair_difference("K", "R")
    assert pair_difference("-", "-") == 0.0


def test_missing_residues_are_excluded_not_counted_as_differences():
    assert pair_difference(MISSING, "K") is None
    partial = discrimination(
        {"mouse": "K", "rat": MISSING, "human": "W", "marmoset": "W"}
    )
    full = discrimination({"mouse": "K", "rat": "K", "human": "W", "marmoset": "W"})
    assert partial == pytest.approx(full, abs=0.05)


def test_ambiguous_codes_are_not_treated_as_identical():
    assert pair_difference("X", "K") is None


def test_pattern_consistency_reaches_one_for_a_clean_pattern():
    assert pattern_consistency(
        {"mouse": "K", "rat": "K", "human": "E", "marmoset": "E"}, BINDERS, NON_BINDERS
    ) == pytest.approx(1.0)


def test_exposure_ramp_is_smooth_and_bounded():
    assert exposure_weight(0.0) == 0.0
    assert exposure_weight(0.05) == 0.0
    assert exposure_weight(0.35) == 1.0
    assert exposure_weight(1.0) == 1.0
    assert 0 < exposure_weight(0.2) < 1
    values = [exposure_weight(x / 100) for x in range(0, 100)]
    assert all(b >= a for a, b in zip(values, values[1:]))  # monotone
    assert exposure_weight(float("nan")) == 0.0


def test_confidence_weight_floors_rather_than_dropping_low_plddt():
    assert confidence_weight(90) == 1.0
    assert confidence_weight(70) == 1.0
    assert confidence_weight(40) == 0.5
    assert 0.5 < confidence_weight(60) < 1.0
    assert confidence_weight(float("nan")) == 1.0  # not an AlphaFold model


def test_composite_keeps_its_three_factors_separable():
    composite, exposure, confidence = composite_score(0.8, 0.5, 60.0)
    assert composite == pytest.approx(0.8 * exposure * confidence)
    assert exposure == 1.0
    assert confidence == 0.75


def _dataset(sequences, calls):
    records = [SpeciesRecord(name=n, sequence=s) for n, s in sequences.items()]
    dataset, _ = build_dataset(records, calls, "mouse")
    setattr(dataset, "_alignment_sequences", sequences)
    return dataset


def test_upgma_root_split_finds_the_two_clades():
    names = ["mouse", "rat", "human", "marmoset"]
    distances = {}
    groups = {"mouse": 0, "rat": 0, "human": 1, "marmoset": 1}
    for a in names:
        for b in names:
            if a != b:
                distances[(a, b)] = 0.05 if groups[a] == groups[b] else 0.4
    left, right = upgma_root_split(names, distances)
    assert {frozenset(left), frozenset(right)} == {
        frozenset({"mouse", "rat"}),
        frozenset({"human", "marmoset"}),
    }


def _scores(sequences, dataset):
    from epitope_map.align import Alignment, ResidueMap
    from epitope_map.score import score_alignment

    alignment = Alignment(sequences=sequences, reference="mouse", method="test")
    keys = {}
    residue_map = ResidueMap(alignment, keys)
    return score_alignment(residue_map, dataset)


def test_two_clade_comparison_is_reported_as_degenerate():
    sequences = {
        "mouse": "AAAAKKKKCCCC",
        "rat": "AAAAKKKKCCCC",
        "human": "AAAADDDDCCCC",
        "marmoset": "AAAADDDDCCCC",
    }
    calls = {"mouse": "binder", "rat": "binder", "human": "non_binder", "marmoset": "non_binder"}
    dataset = _dataset(sequences, calls)
    report = assess_degeneracy(_scores(sequences, dataset), dataset)
    assert report.clade_split is True
    assert report.is_degenerate
    assert "top-level split" in report.clade_split_detail


def test_pattern_cutting_across_the_tree_is_not_degenerate():
    sequences = {
        "mouse": "AAAAKKKKCCCC",
        "rat": "AAAAKKKKCCCG",
        "human": "AAAAKKKKCCCW",
        "marmoset": "AAAADDDDCCCC",
    }
    # binder/non-binder assignment deliberately crosses the rodent/primate split
    calls = {"mouse": "binder", "human": "binder", "rat": "non_binder", "marmoset": "non_binder"}
    dataset = _dataset(sequences, calls)
    report = assess_degeneracy(_scores(sequences, dataset), dataset)
    assert report.clade_split is False


def test_background_rate_is_reported_for_calibration():
    sequences = {
        "mouse": "AAAAKKKKCCCC",
        "rat": "AAAAKKKKCCCC",
        "human": "AAAADDDDCCCC",
        "marmoset": "AAAADDDDCCCC",
    }
    calls = {"mouse": "binder", "rat": "binder", "human": "non_binder", "marmoset": "non_binder"}
    dataset = _dataset(sequences, calls)
    report = assess_degeneracy(_scores(sequences, dataset), dataset)
    assert 0.0 <= report.observed_fraction <= 1.0
    assert report.background_fraction == report.background_fraction  # not NaN
    assert report.n_labelings > 0
    assert report.exhaustive is True


def test_indel_in_a_buried_helix_scores_below_one_in_an_exposed_coil():
    """Regression test 5: same length, opposite geometry, opposite verdict."""
    from epitope_map.score import indel_score

    coil = indel_score(2, "-", rsa=0.75, flank_rsa=0.6, distance_to_patch=12.0)
    helix = indel_score(2, "H", rsa=0.05, flank_rsa=0.05, distance_to_patch=12.0)
    assert coil > helix
    assert coil > 0.5
    assert helix < 0.1


def test_a_single_residue_surface_insertion_is_not_crushed():
    """The overcorrection: a real rodent-specific insertion fell to 0.25."""
    from epitope_map.score import indel_score, indel_weight

    # the sequence signal is only damped for the artifact case, not for length
    assert indel_weight(1, "-") == 1.0
    assert indel_weight(3, "-") == 1.0
    assert indel_weight(1, "H") < 1.0
    # and the geometric score still rates a lone exposed insertion as real
    assert indel_score(1, "-", rsa=0.86, flank_rsa=0.6, distance_to_patch=14.0) >= 0.5


def test_indel_score_rewards_length_exposure_and_proximity():
    from epitope_map.score import indel_score

    base = dict(secondary_structure="-", rsa=0.6, flank_rsa=0.6, distance_to_patch=10.0)
    assert indel_score(length=3, **base) > indel_score(length=1, **base)
    exposed = indel_score(2, "-", rsa=0.8, flank_rsa=0.8, distance_to_patch=10.0)
    buried = indel_score(2, "-", rsa=0.02, flank_rsa=0.02, distance_to_patch=10.0)
    assert exposed > buried
    near = indel_score(2, "-", rsa=0.6, flank_rsa=0.6, distance_to_patch=10.0)
    far = indel_score(2, "-", rsa=0.6, flank_rsa=0.6, distance_to_patch=120.0)
    assert near > far
    assert 0.0 <= far <= 1.0
