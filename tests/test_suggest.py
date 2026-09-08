"""Chimera boundary and reciprocal mutant suggestions."""

import pytest


def test_both_mutant_directions_are_generated(synthetic_result):
    directions = {m.direction for m in synthetic_result.mutants}
    assert directions == {"loss_of_binding", "gain_of_binding"}


def test_gain_of_binding_mutants_are_numbered_in_the_non_binder_background(
    synthetic_result,
):
    gain = [m for m in synthetic_result.mutants if m.direction == "gain_of_binding"]
    assert gain
    for mutant in gain:
        assert mutant.background != synthetic_result.alignment.reference
        assert mutant.background in {r.name for r in synthetic_result.dataset.non_binders}
        # the wild type is the non-binder residue and the mutation restores the
        # reference residue
        sequence = synthetic_result.dataset.get(mutant.background).sequence
        assert sequence[mutant.position - 1] == mutant.wild_type


def test_loss_of_binding_mutants_are_numbered_in_the_reference(synthetic_result):
    reference = synthetic_result.alignment.reference
    sequence = synthetic_result.dataset.get(reference).sequence
    loss = [m for m in synthetic_result.mutants if m.direction == "loss_of_binding"]
    assert loss
    for mutant in loss:
        assert mutant.background == reference
        assert sequence[mutant.position - 1] == mutant.wild_type


def test_gain_of_binding_is_prioritised_over_the_same_loss_mutant(synthetic_result):
    pairs = {}
    for mutant in synthetic_result.mutants:
        key = (mutant.patch_id, mutant.ref_number)
        pairs.setdefault(key, {})[mutant.direction] = mutant.priority
    both = [v for v in pairs.values() if len(v) == 2]
    assert both
    assert all(v["gain_of_binding"] > v["loss_of_binding"] for v in both)


def test_mutants_are_ranked_by_grantham_and_exposure(synthetic_result):
    top_patch = synthetic_result.patches[0].patch_id
    mutants = [m for m in synthetic_result.mutants if m.patch_id == top_patch]
    priorities = [m.priority for m in mutants]
    assert priorities == sorted(priorities, reverse=True)
    assert all(m.grantham >= 0 for m in mutants)


def test_chimera_segments_cover_every_patch_member(synthetic_result):
    by_patch = {p.patch_id: p for p in synthetic_result.patches}
    assert synthetic_result.chimeras
    for chimera in synthetic_result.chimeras:
        patch = by_patch[chimera.patch_id]
        for member in patch.members:
            assert any(
                segment.start_ref_index <= member.ref_index <= segment.end_ref_index
                for segment in chimera.segments
            )
        assert chimera.length == sum(s.length for s in chimera.segments)


def test_discontinuous_patch_is_split_into_separate_swaps(synthetic_result):
    """A patch with members far apart in sequence gets one segment per run."""
    top = synthetic_result.patches[0]
    indices = sorted(m.ref_index for m in top.members)
    chimera = next(c for c in synthetic_result.chimeras if c.patch_id == top.patch_id)
    if max(indices) - min(indices) > 20:
        assert len(chimera.segments) > 1
        # and the swap is far smaller than the span it covers
        assert chimera.length < (max(indices) - min(indices) + 1)


def test_chimera_says_so_when_dssp_is_unavailable(synthetic_result):
    if synthetic_result.structure.dssp_used:
        pytest.skip("DSSP is installed in this environment")
    assert all(not c.ss_respected for c in synthetic_result.chimeras)
    assert all("DSSP unavailable" in c.note for c in synthetic_result.chimeras)


def test_chimera_avoids_cutting_secondary_structure_when_dssp_is_available(
    synthetic_result,
):
    """With DSSP assignments present, boundaries move out to coil/turn."""
    from epitope_map.suggest import suggest_chimeras

    structure = synthetic_result.structure
    residues = synthetic_result.residues
    # simulate DSSP output: a helix covering the first top patch
    patch = synthetic_result.patches[0]
    span = range(patch.members[0].ref_index - 3, patch.members[0].ref_index + 4)
    saved = {r.ref_index: r.secondary_structure for r in residues}
    for residue in residues:
        residue.secondary_structure = "H" if residue.ref_index in span else "-"
    structure.dssp_used = True
    try:
        chimeras = suggest_chimeras(
            [patch], residues, synthetic_result.residue_map, structure, top_n=1
        )
    finally:
        structure.dssp_used = False
        for residue in residues:
            residue.secondary_structure = saved[residue.ref_index]

    chimera = chimeras[0]
    assert chimera.ss_respected
    assert chimera.start_ref_index < patch.members[0].ref_index  # pushed out of the helix
    assert "DSSP" in chimera.note
