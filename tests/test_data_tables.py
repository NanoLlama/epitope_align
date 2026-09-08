"""Grantham distances and Tien maximum-ASA values."""

import pytest

from epitope_map.data.grantham import grantham_distance, normalized_grantham
from epitope_map.data.max_asa import MAX_ASA_TIEN, relative_sasa

# published Grantham (1974) values; the reconstruction is allowed +/-2
PUBLISHED = {
    ("C", "W"): 215, ("G", "W"): 184, ("S", "R"): 110, ("L", "I"): 5,
    ("F", "Y"): 22, ("D", "E"): 45, ("K", "R"): 26, ("A", "G"): 60,
    ("A", "V"): 64, ("C", "G"): 159, ("W", "V"): 88, ("M", "L"): 15,
}


@pytest.mark.parametrize("pair,expected", sorted(PUBLISHED.items()))
def test_matches_published_matrix(pair, expected):
    assert abs(grantham_distance(*pair) - expected) <= 2


def test_symmetric_and_zero_on_the_diagonal():
    for a in "ACDEFGHIKLMNPQRSTVWY":
        assert grantham_distance(a, a) == 0
        for b in "ACDEFGHIKLMNPQRSTVWY":
            assert grantham_distance(a, b) == grantham_distance(b, a)


def test_normalisation_bounds():
    assert normalized_grantham("A", "A") == 0.0
    assert 0.98 <= normalized_grantham("C", "W") <= 1.0
    assert all(
        0.0 <= normalized_grantham(a, b) <= 1.0
        for a in "ACDEFGHIKLMNPQRSTVWY"
        for b in "ACDEFGHIKLMNPQRSTVWY"
    )


def test_unknown_codes_are_nan_not_zero():
    value = grantham_distance("X", "A")
    assert value != value  # NaN: unknown, not "identical"


def test_max_asa_table_is_complete():
    assert set(MAX_ASA_TIEN) == set("ACDEFGHIKLMNPQRSTVWY")
    assert MAX_ASA_TIEN["G"] < MAX_ASA_TIEN["W"]


def test_relative_sasa():
    assert relative_sasa("A", 0.0) == 0.0
    assert relative_sasa("A", 129.0) == pytest.approx(1.0)
    assert relative_sasa("Z", 100.0) == pytest.approx(0.5)  # default max ASA
    assert relative_sasa("A", 1e6) <= 1.5  # clamped
