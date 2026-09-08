"""Clustering, patch metrics and ranking."""

import pytest

from epitope_map.patches import (
    MAX_PLAUSIBLE_SPREAD,
    baseline_counts,
    find_patches,
)
from epitope_map.score import ResidueAnalysis


def residue(index, xyz, discrimination=0.8, composite=None, **kwargs):
    row = ResidueAnalysis(
        ref_index=index,
        column=index,
        aa="K",
        ref_number=str(index + 1),
        discrimination=discrimination,
        modelled=True,
        centroid=xyz,
        rsa=0.5,
        **kwargs,
    )
    row.composite = discrimination if composite is None else composite
    return row


def test_two_separated_clusters_are_kept_apart():
    residues = [residue(i, (i * 4.0, 0.0, 0.0)) for i in range(3)]
    residues += [residue(10 + i, (100.0 + i * 4.0, 0.0, 0.0)) for i in range(3)]
    patches, singletons = find_patches(residues, discrimination_cutoff=0.25, radius=12.0)
    assert len(patches) == 2
    assert all(p.size == 3 for p in patches)
    assert not singletons


def test_singletons_are_reported_not_dropped():
    residues = [residue(i, (i * 4.0, 0.0, 0.0)) for i in range(2)]
    residues.append(residue(9, (80.0, 0.0, 0.0)))
    patches, singletons = find_patches(residues, discrimination_cutoff=0.25, radius=12.0)
    assert len(patches) == 1
    assert len(singletons) == 1
    assert singletons[0].size == 1


def test_masked_and_buried_residues_do_not_seed_patches():
    residues = [residue(i, (i * 4.0, 0.0, 0.0)) for i in range(3)]
    residues[1].mask_reasons.append("buried")
    residues[2].mask_reasons.append("outside_ectodomain")
    patches, singletons = find_patches(residues, discrimination_cutoff=0.25, radius=12.0)
    assert not patches
    assert len(singletons) == 1


def test_residues_below_the_cutoff_are_excluded():
    residues = [residue(i, (i * 4.0, 0.0, 0.0), discrimination=0.1) for i in range(4)]
    patches, singletons = find_patches(residues, discrimination_cutoff=0.25, radius=12.0)
    assert not patches and not singletons


def test_diffuse_patch_is_flagged_and_loses_the_normalised_ranking():
    diffuse = [residue(i, (i * 10.0, 0.0, 0.0), discrimination=0.6) for i in range(6)]
    tight = [
        residue(100 + i, (500.0 + i * 3.0, 0.0, 0.0), discrimination=0.9)
        for i in range(3)
    ]
    patches, _ = find_patches(diffuse + tight, discrimination_cutoff=0.25, radius=12.0)
    by_id = {p.patch_id: p for p in patches}
    big = max(by_id.values(), key=lambda p: p.size)
    small = min(by_id.values(), key=lambda p: p.size)
    assert big.total_score > small.total_score      # raw ranking favours the big one
    assert big.rank_raw == 1
    assert small.rank_normalized == 1               # normalised ranking does not
    assert big.spread > MAX_PLAUSIBLE_SPREAD
    assert any("spread" in flag for flag in big.flags)
    assert small.compactness == 1.0
    assert big.compactness < 1.0


def test_dbscan_does_not_chain_through_a_single_bridge():
    """DBSCAN border points do not extend a cluster; connected components do.

    ``min_patch_size`` doubles as DBSCAN's ``min_samples``, so the lone bridging
    residue is a border point rather than a core point.
    """
    left = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(4)]
    bridge = [residue(50, (20.0, 0.0, 0.0))]
    right = [residue(60 + i, (31.0 + i * 3.0, 0.0, 0.0)) for i in range(4)]
    residues = left + bridge + right
    graph_patches, _ = find_patches(
        residues, discrimination_cutoff=0.25, radius=12.0, method="graph", min_size=4
    )
    dbscan_patches, _ = find_patches(
        residues, discrimination_cutoff=0.25, radius=12.0, method="dbscan", min_size=4
    )
    assert len(graph_patches) == 1              # single linkage chains through
    assert len(dbscan_patches) == 2             # DBSCAN keeps the two surfaces apart


def test_patch_metrics():
    residues = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(4)]
    residues[0].max_grantham = 180.0
    residues[1].involves_gap = True
    residues[2].glycan_flags.append("near sequon")
    residues[3].sasa = 100.0
    patches, _ = find_patches(residues, discrimination_cutoff=0.25, radius=12.0)
    patch = patches[0]
    assert patch.size == 4
    assert patch.max_grantham == 180.0
    assert patch.n_indels == 1
    assert patch.n_glycan_flagged == 1
    assert patch.spread == pytest.approx(9.0)
    assert patch.centroid[0] == pytest.approx(4.5)
    assert patch.surface_area == pytest.approx(100.0)
    assert "Grantham" in patch.rationale()


def test_baseline_counts_track_each_narrowing_step():
    residues = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(5)]
    residues[0].discrimination = 0.0
    residues[1].buried = True
    residues[1].mask_reasons.append("buried")
    residues[2].in_ectodomain = False
    counts = baseline_counts(residues, 0.25)
    assert counts["all_reference_residues"] == 5
    assert counts["in_ectodomain"] == 4
    assert counts["discriminating"] == 3
    assert counts["discriminating_and_exposed"] == 2
    assert counts["after_context_masking"] == 2
