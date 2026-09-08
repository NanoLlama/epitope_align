"""Clustering, patch metrics and ranking."""

import math

import pytest

from epitope_map.patches import (
    MAX_PLAUSIBLE_SPREAD,
    baseline_counts,
    find_patches,
    membership_hash,
    merged_surfaces,
    promote_singletons,
    radius_sweep,
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
    residues[2].accessible = False
    counts = baseline_counts(residues, 0.25)
    assert counts["all_reference_residues"] == 5
    assert counts["reachable"] == 4
    assert counts["discriminating"] == 3
    assert counts["discriminating_and_exposed"] == 2
    assert counts["after_context_masking"] == 2


def test_min_distance_beats_centroid_separation_for_elongated_patches():
    """Two long patches can be far apart by centroid and touching in fact."""
    left = [residue(i, (i * 4.0, 0.0, 0.0)) for i in range(6)]     # 0..20 on x
    right = [residue(20 + i, (26.0 + i * 4.0, 0.0, 0.0)) for i in range(6)]
    patches, _ = find_patches(left + right, discrimination_cutoff=0.25, radius=5.0)
    assert len(patches) == 2
    a, b = patches
    centroid_gap = math.dist(a.centroid, b.centroid)
    assert a.min_distance_to(b) == pytest.approx(6.0)
    assert centroid_gap > 3 * a.min_distance_to(b)
    assert any(pid == b.patch_id for pid, _ in a.neighbours)


def test_groups_20A_apart_separate_at_12A_and_merge_at_18A():
    """Regression test 3 from the change request."""
    left = [residue(i, (i * 4.0, 0.0, 0.0)) for i in range(4)]        # x = 0..12
    right = [residue(20 + i, (32.0 + i * 4.0, 0.0, 0.0)) for i in range(4)]  # gap 20
    residues = left + right

    at12, _ = find_patches(residues, discrimination_cutoff=0.25, radius=12.0)
    assert len(at12) == 2

    rows = radius_sweep(residues, discrimination_cutoff=0.25, radii=(10, 12, 14, 16, 18, 20))
    by_radius = {row["radius_A"]: row for row in rows}
    assert by_radius[12]["n_patches"] == 2
    assert by_radius[20]["n_patches"] == 1
    assert by_radius[20]["largest_patch"] == 8
    # the sweep must not disturb the patch assignment of the real run
    assert {r.patch_id for r in residues} == {p.patch_id for p in at12} | {None} - {None}


def test_merged_surfaces_report_the_combined_area():
    left = [residue(i, (i * 4.0, 0.0, 0.0), composite=0.5) for i in range(3)]
    right = [residue(20 + i, (22.0 + i * 4.0, 0.0, 0.0), composite=0.5) for i in range(3)]
    for r in left + right:
        r.sasa = 100.0
    patches, singles = find_patches(left + right, discrimination_cutoff=0.25, radius=12.0)
    assert len(patches) == 2
    surfaces = merged_surfaces(patches, singles)
    assert len(surfaces) == 1
    surface = surfaces[0]
    assert surface["n_residues"] == 6
    assert surface["accessible_area_A2"] == pytest.approx(600.0)
    assert set(surface["patches"]) == {p.patch_id for p in patches}


def test_patches_carry_conserved_surface_context_without_scoring_it():
    members = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(3)]
    conserved = residue(50, (4.0, 3.0, 0.0), discrimination=0.0, composite=0.0)
    patches, _ = find_patches(
        members + [conserved], discrimination_cutoff=0.25, radius=12.0
    )
    patch = patches[0]
    assert patch.size == 3                       # score comes from members only
    assert [r.ref_index for r in patch.context] == [50]
    assert patch.n_total_surface == 4
    assert patch.total_score == pytest.approx(sum(m.composite for m in members))


def test_isolated_indel_is_promoted_not_buried():
    """A lone surface insertion outranks a dump of leftovers."""
    cluster = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(3)]
    insertion = residue(40, (100.0, 0.0, 0.0), discrimination=0.9, composite=0.9)
    insertion.involves_gap = True
    dull = residue(60, (300.0, 0.0, 0.0), discrimination=0.3, composite=0.3)
    patches, singletons = find_patches(
        cluster + [insertion, dull], discrimination_cutoff=0.25, radius=12.0
    )
    high, low = promote_singletons(singletons, patches)
    assert [p.members[0].ref_index for p in high] == [40]
    assert "indel" in high[0].promoted_reason
    assert [p.members[0].ref_index for p in low] == [60]


def test_singleton_near_a_ranked_patch_is_promoted():
    cluster = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(3)]
    nearby = residue(40, (24.0, 0.0, 0.0), discrimination=0.3, composite=0.3)
    patches, singletons = find_patches(
        cluster + [nearby], discrimination_cutoff=0.25, radius=12.0
    )
    high, low = promote_singletons(singletons, patches)
    assert [p.members[0].ref_index for p in high] == [40]
    assert "plausibly part of" in high[0].promoted_reason
    assert not low


def test_patch_ids_are_derived_from_content_not_rank():
    """Regression test 2: 'patch PD' must mean the same residues after a re-run."""
    strong = [residue(200 + i, (i * 3.0, 0.0, 0.0), composite=0.9) for i in range(3)]
    weak = [residue(50 + i, (100.0 + i * 3.0, 0.0, 0.0), composite=0.2) for i in range(3)]

    first, _ = find_patches(strong + weak, discrimination_cutoff=0.25, radius=12.0)
    again, _ = find_patches(
        [
            residue(r.ref_index, r.centroid, composite=r.composite)
            for r in strong + weak
        ],
        discrimination_cutoff=0.25,
        radius=12.0,
    )
    assert [p.patch_id for p in first] == [p.patch_id for p in again]
    # named after the lowest member, so the name says where the patch is
    assert {p.patch_id for p in first} == {"p:51", "p:201"}

    # the low-scoring patch keeps its identity even though it outranks nothing
    by_id = {p.patch_id: sorted(m.ref_index for m in p.members) for p in first}
    assert by_id["p:51"] == [50, 51, 52]


def test_changing_one_patch_leaves_the_others_identified_the_same():
    base = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(3)]
    far = [residue(60 + i, (100.0 + i * 3.0, 0.0, 0.0)) for i in range(3)]
    before, _ = find_patches(base + far, discrimination_cutoff=0.25, radius=12.0)

    extra = residue(63, (109.0, 0.0, 0.0))
    after, _ = find_patches(base + far + [extra], discrimination_cutoff=0.25, radius=12.0)

    before_ids = {p.patch_id: membership_hash(p.members) for p in before}
    after_ids = {p.patch_id: membership_hash(p.members) for p in after}
    assert set(before_ids) == set(after_ids)  # both patches keep their names
    # the changed patch's membership digest moves; the untouched one does not
    assert after_ids["p:61"] != before_ids["p:61"]
    assert after_ids["p:1"] == before_ids["p:1"]


def test_membership_hash_ignores_ordering():
    a = [residue(i, (i * 3.0, 0.0, 0.0)) for i in range(3)]
    assert membership_hash(a) == membership_hash(list(reversed(a)))
    assert membership_hash(a) != membership_hash(a[:2])


def _group(start_index, origin, n=3, spacing=3.0):  # noqa: D401
    return [
        residue(start_index + i, (origin[0] + i * spacing, origin[1], origin[2]))
        for i in range(n)
    ]


def test_merging_grows_by_diameter_not_by_chaining():
    """Regression test 3: 15 A, 15 A and 60 A apart is two surfaces, not one."""
    a = _group(0, (0.0, 0.0, 0.0))
    b = _group(20, (21.0, 0.0, 0.0))     # ~15 A from a's nearest member
    c = _group(40, (42.0, 0.0, 0.0))     # ~15 A from b, but 42 A from a
    far = _group(60, (150.0, 0.0, 0.0))  # 60+ A from everything

    patches, singles = find_patches(
        a + b + c + far, discrimination_cutoff=0.25, radius=12.0
    )
    assert len(patches) == 4

    surfaces = merged_surfaces(patches, singles, footprint_diameter=30.0)
    assert surfaces, "neighbouring patches should still group"
    # nothing may span the whole chain: single linkage would have merged a-b-c
    for surface in surfaces:
        assert float(surface["spread_A"]) <= 30.0
        assert surface["verdict"] == "plausible single epitope"
    # and the isolated group is never pulled in
    assert all("p:61" not in surface["patches"] for surface in surfaces)
    # alternative overlapping groupings are offered, not one verdict
    assert len(surfaces) >= 2


def test_merged_area_is_the_union_not_a_sum():
    a = _group(0, (0.0, 0.0, 0.0), n=2)
    b = _group(20, (18.0, 0.0, 0.0), n=2)  # 15 A gap: separate patches, one surface
    for r in a + b:
        r.sasa = 100.0
    patches, singles = find_patches(a + b, discrimination_cutoff=0.25, radius=12.0)
    surfaces = merged_surfaces(patches, singles, footprint_diameter=30.0)
    assert surfaces[0]["n_residues"] == 4
    assert surfaces[0]["accessible_area_A2"] == pytest.approx(400.0)


def test_a_merge_that_would_breach_the_footprint_is_refused():
    a = _group(0, (0.0, 0.0, 0.0), n=2)
    # 25 A apart, so they are neighbours, but a 31 A union breaches the footprint
    b = _group(20, (28.0, 0.0, 0.0), n=2)
    patches, singles = find_patches(a + b, discrimination_cutoff=0.25, radius=12.0)
    surfaces = merged_surfaces(patches, singles, footprint_diameter=30.0)
    assert not surfaces


def test_a_wholly_glycan_proximal_patch_is_discounted_and_labelled():
    """P2-7: that patch is a glycan hypothesis, not a surface hypothesis."""
    from epitope_map.patches import GLYCAN_PENALTY, apply_confidence_penalties

    members = _group(0, (0.0, 0.0, 0.0))
    for r in members:
        r.glycan_flags.append("within 12 A of a differential sequon")
    clean = _group(60, (200.0, 0.0, 0.0))

    patches, _ = find_patches(members + clean, discrimination_cutoff=0.25, radius=12.0)
    apply_confidence_penalties(patches, oligomer_unmodelled=False)

    glycan_patch = next(p for p in patches if p.members[0].ref_index == 0)
    surface_patch = next(p for p in patches if p.members[0].ref_index == 60)
    assert glycan_patch.confidence_penalty == GLYCAN_PENALTY
    assert glycan_patch.total_score < glycan_patch.raw_total_score
    assert any("glycan hypothesis" in flag for flag in glycan_patch.flags)
    assert surface_patch.confidence_penalty == 1.0
    assert surface_patch.total_score == surface_patch.raw_total_score


def test_a_patch_on_an_unmodelled_interface_is_discounted():
    from epitope_map.patches import OLIGOMER_PENALTY, apply_confidence_penalties

    members = _group(700, (0.0, 0.0, 0.0))
    patches, _ = find_patches(members, discrimination_cutoff=0.25, radius=12.0)
    apply_confidence_penalties(
        patches, oligomer_unmodelled=True, interface_regions=[(690, 720)]
    )
    assert patches[0].confidence_penalty == OLIGOMER_PENALTY
    assert any("oligomerisation" in flag for flag in patches[0].flags)

    # supplying the assembly removes the discount
    patches, _ = find_patches(members, discrimination_cutoff=0.25, radius=12.0)
    apply_confidence_penalties(
        patches, oligomer_unmodelled=False, interface_regions=[(690, 720)]
    )
    assert patches[0].confidence_penalty == 1.0


def test_penalties_change_the_ranking_not_just_the_prose():
    from epitope_map.patches import _rank, apply_confidence_penalties

    strong = _group(0, (0.0, 0.0, 0.0), n=4)
    for r in strong:
        r.composite = 0.6
        r.glycan_flags.append("near a sequon")
    weaker = _group(60, (200.0, 0.0, 0.0), n=4)
    for r in weaker:
        r.composite = 0.45

    patches, _ = find_patches(strong + weaker, discrimination_cutoff=0.25, radius=12.0)
    assert patches[0].members[0].ref_index == 0  # the glycan patch leads on raw score

    apply_confidence_penalties(patches, oligomer_unmodelled=False)
    _rank(patches)
    assert patches[0].members[0].ref_index == 60  # and loses it once discounted
