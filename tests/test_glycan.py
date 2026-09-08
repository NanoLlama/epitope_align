"""N-glycosylation sequon scanning and glycan proximity flags."""

from epitope_map.glycan import scan_sequons


def test_sequon_motif_rules():
    assert scan_sequons("AANGTA") == [2]        # N-G-T
    assert scan_sequons("AANPTA") == []         # X = P is not a sequon
    assert scan_sequons("AANGSA") == [2]        # S also counts
    assert scan_sequons("AANGAA") == []         # third position must be S or T
    assert scan_sequons("AAAAAN") == []         # truncated at the C terminus


def test_gaps_neither_create_nor_break_a_sequon():
    # the gap sits inside the motif: on the ungapped sequence this is still N-G-T
    assert scan_sequons("AAN-GTA") == [2]
    # gaps are invisible to the scan: this species really does read N-A-T
    assert scan_sequons("AAN--A-T") == [2]
    # and a proline hidden behind a gap still disqualifies the motif
    assert scan_sequons("AAN-PT") == []


def test_differential_sequon_and_proximity_flags(synthetic_result):
    glycans = synthetic_result.glycans
    assert glycans.sequons
    assert glycans.differential, "the synthetic panel plants a primate-only sequon"
    sequon = glycans.differential[0]
    assert set(sequon.present_in) == {"human", "marmoset", "macaque"}
    assert set(sequon.absent_in) == {"mouse", "rat", "hamster"}

    flagged = [r for r in synthetic_result.residues if r.glycan_flags]
    assert flagged, "residues near a differential sequon should be flagged"
    assert all("within" in flag for r in flagged for flag in r.glycan_flags)
    assert any(r.is_sequon_asn for r in synthetic_result.residues)


def test_report_warns_that_glycans_act_at_a_distance(synthetic_result):
    text = " ".join(synthetic_result.glycans.warnings)
    assert "at a distance" in text
    assert "outside the antibody footprint" in text
