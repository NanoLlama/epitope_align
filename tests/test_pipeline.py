"""End-to-end behaviour, including the deliberate degenerate case."""

import csv
from pathlib import Path

import pytest

from epitope_map import demo as synthetic
from epitope_map.io_seq import InputError
from epitope_map.pipeline import RunConfig, run_pipeline
from epitope_map.topology import TopologyUnknown
from epitope_map.report import write_all


def test_planted_epitope_is_the_top_patch(synthetic_result):
    """Positive control: the planted cap should come out ranked first."""
    truth = set(synthetic.truth_numbers())
    top = synthetic_result.patches[0]
    found = {m.ref_number for m in top.members}
    assert top.rank_raw == 1
    assert top.rank_normalized == 1
    assert truth <= found, f"missing planted residues: {truth - found}"
    assert len(found - truth) <= 2, "top patch should not be padded with noise"


def test_narrowing_beats_the_naive_baseline(synthetic_result):
    counts = synthetic_result.counts
    top = synthetic_result.patches[0]
    naive = counts["discriminating_and_exposed"]
    assert top.size < naive  # the structural filters actually narrow things
    assert counts["discriminating"] < counts["all_reference_residues"]


def test_buried_residues_are_kept_but_masked(synthetic_result):
    buried = [r for r in synthetic_result.residues if r.buried]
    assert buried, "the packed core should be buried"
    assert all("buried" in r.mask_reasons for r in buried if r.modelled)
    assert all(r.patch_id is None for r in buried)
    # still present in the output table, not deleted
    assert len(synthetic_result.residues) == len(synthetic_result.residue_map)


def test_unmodelled_residues_survive_into_the_table(synthetic_result):
    unmodelled = [r for r in synthetic_result.residues if not r.modelled]
    assert len(unmodelled) == len(synthetic.UNMODELLED_INDICES)
    assert all("not_modelled_in_structure" in r.mask_reasons for r in unmodelled)
    assert all(r.discrimination >= 0 for r in unmodelled)


def test_two_clade_run_reports_weak_discrimination(synthetic_result):
    """The degenerate case must be announced, not dressed up as confidence."""
    degeneracy = synthetic_result.degeneracy
    assert degeneracy.clade_split is True
    assert degeneracy.is_degenerate
    assert any("TWO-CLADE DEGENERACY" in w for w in synthetic_result.warnings)
    assert degeneracy.background_fraction == degeneracy.background_fraction


def test_outputs_are_all_written(synthetic_result, tmp_path):
    paths = write_all(synthetic_result, tmp_path)
    assert set(paths) == {
        "residues", "patches", "chimeras", "mutants", "alignment", "pymol", "report"
    }
    for path in paths.values():
        assert path.exists() and path.stat().st_size > 0

    with open(paths["residues"]) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == len(synthetic_result.residues)
    for column in (
        "ref_number", "discrimination", "rsa", "exposure_weight",
        "confidence_weight", "composite", "patch_id", "mask_reasons",
    ):
        assert column in rows[0]
    # the three composite factors are separable, as the spec requires
    row = next(r for r in rows if r["composite"] not in ("", "0.0"))
    assert float(row["composite"]) == pytest.approx(
        float(row["discrimination"])
        * float(row["exposure_weight"])
        * float(row["confidence_weight"]),
        abs=1e-3,
    )


def test_report_states_the_caveats(synthetic_result, tmp_path):
    paths = write_all(synthetic_result, tmp_path)
    text = paths["report"].read_text()
    for phrase in (
        "not a prediction",
        "need not lie inside the antibody footprint",
        "energetically",
        "Two-clade comparisons have limited resolving power",
        "AlphaFold surfaces are unglycosylated",
    ):
        assert phrase in text
    assert "session.pml" in text


def test_pymol_script_references_the_top_patches(synthetic_result, tmp_path):
    paths = write_all(synthetic_result, tmp_path)
    text = paths["pymol"].read_text()
    assert "spectrum b, white_red" in text
    for patch in synthetic_result.patches[: synthetic_result.config.top_n]:
        assert f"select {patch.patch_id}," in text


def test_ectodomain_range_restricts_the_analysis(synthetic_inputs, tmp_path):
    config = RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(synthetic_inputs["binding"]),
        reference="mouse",
        structure=str(synthetic_inputs["structure"]),
        topology="whole-chain",
        outdir=tmp_path,
        ectodomain=(60, 100),
    )
    result = run_pipeline(config)
    inside = [r for r in result.residues if r.in_ectodomain]
    assert all(60 <= int(r.ref_number.rstrip("ABC")) <= 100 for r in inside)
    assert result.counts["in_ectodomain"] == len(inside)
    assert all(
        60 <= int(m.ref_number.rstrip("ABC")) <= 100
        for patch in result.patches
        for m in patch.members
    )


def test_unknown_species_is_aligned_but_not_scored(synthetic_inputs, tmp_path):
    binding = tmp_path / "binding.csv"
    original = Path(synthetic_inputs["binding"]).read_text()
    binding.write_text(original.replace("hamster,binder", "hamster,unknown"))
    config = RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(binding),
        reference="mouse",
        structure=str(synthetic_inputs["structure"]),
        topology="whole-chain",
        outdir=tmp_path,
    )
    result = run_pipeline(config)
    assert "hamster" in result.alignment.sequences          # aligned
    assert "hamster" not in {r.name for r in result.dataset.scored}  # not scored
    assert result.degeneracy.n_binders == 2


def test_missing_non_binder_is_rejected(synthetic_inputs, tmp_path):
    binding = tmp_path / "binding.csv"
    binding.write_text(
        "species,binding\nmouse,binder\nrat,binder\nhamster,binder\n"
        "human,unknown\nmarmoset,unknown\nmacaque,unknown\n"
    )
    config = RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(binding),
        reference="mouse",
        structure=str(synthetic_inputs["structure"]),
        topology="whole-chain",
        outdir=tmp_path,
    )
    with pytest.raises(InputError, match="non_binder"):
        run_pipeline(config)


def test_an_informative_species_shrinks_the_candidate_set(tmp_path, synthetic_result):
    """The report's standing advice, checked rather than asserted."""
    inputs = synthetic.write_inputs(tmp_path / "extra", include_informative=True)
    config = RunConfig(
        sequences=str(inputs["sequences"]),
        binding=str(inputs["binding"]),
        reference="mouse",
        structure=str(inputs["structure"]),
        topology="whole-chain",
        outdir=tmp_path / "out",
    )
    result = run_pipeline(config)
    assert result.degeneracy.clade_split is False  # the pattern now cuts the tree
    assert (
        result.counts["discriminating_and_exposed"]
        < synthetic_result.counts["discriminating_and_exposed"]
    )
    # and the planted epitope is still recovered
    truth = set(synthetic.truth_numbers())
    assert truth <= {m.ref_number for m in result.patches[0].members}


def _structure_with_partner(inputs, tmp_path):
    """Copy the synthetic structure and park a partner chain over the epitope."""
    import math

    coords = synthetic.coordinates()
    lines = [
        line
        for line in Path(inputs["structure"]).read_text().splitlines()
        if not line.startswith("END")
    ]
    serial = 90000
    for offset, index in enumerate(synthetic.EPITOPE_INDICES):
        x, y, z = coords[index]
        length = math.sqrt(x * x + y * y + z * z) or 1.0
        # 5 A straight out from the residue, i.e. right on top of it
        px, py, pz = (x / length * (length + 5.0), y / length * (length + 5.0),
                      z / length * (length + 5.0))
        for name, dx in (("N", -1.2), ("CA", 0.0), ("C", 1.2), ("O", 1.6)):
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA B{offset + 1:4d}    "
                f"{px + dx:8.3f}{py:8.3f}{pz:8.3f}  1.00 50.00          {name[0]:>2s}"
            )
            serial += 1
    lines.append("END")
    path = tmp_path / "with_partner.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_partner_chain_masks_the_residues_it_covers(synthetic_inputs, tmp_path):
    structure = _structure_with_partner(synthetic_inputs, tmp_path)
    config = RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(synthetic_inputs["binding"]),
        reference="mouse",
        structure=str(structure),
        chain="A",
        occluding_chains=["B"],
        topology="whole-chain",
        outdir=tmp_path / "out",
    )
    result = run_pipeline(config)
    masked = {
        r.ref_number for r in result.residues if "occluded_by_partner" in r.mask_reasons
    }
    truth = set(synthetic.truth_numbers())
    assert truth & masked, "the partner sits directly on the planted epitope"
    # and masked residues no longer seed patches
    assert not (masked & {m.ref_number for p in result.patches for m in p.members})
    assert any("occluded by the supplied interacting partner" in w for w in result.warnings)


def test_assembly_context_flags_interface_residues_end_to_end(synthetic_inputs, tmp_path):
    structure = _structure_with_partner(synthetic_inputs, tmp_path)
    config = RunConfig(
        sequences=str(synthetic_inputs["sequences"]),
        binding=str(synthetic_inputs["binding"]),
        reference="mouse",
        structure=str(structure),
        chain="A",
        context_chains=["B"],
        topology="whole-chain",
        outdir=tmp_path / "out",
    )
    result = run_pipeline(config)
    interface = {
        r.ref_number for r in result.residues if "assembly_interface" in r.mask_reasons
    }
    assert interface & set(synthetic.truth_numbers())


def _config(inputs, tmp_path, **kwargs):
    base = dict(
        sequences=str(inputs["sequences"]),
        binding=str(inputs["binding"]),
        reference="mouse",
        structure=str(inputs["structure"]),
        outdir=tmp_path / "out",
    )
    base.update(kwargs)
    return RunConfig(**base)


def test_run_refuses_when_topology_is_unknown(synthetic_inputs, tmp_path):
    """Defaulting to the whole chain ranks residues an antibody cannot reach."""
    with pytest.raises(TopologyUnknown) as excinfo:
        run_pipeline(_config(synthetic_inputs, tmp_path))
    message = str(excinfo.value)
    assert "--topology" in message and "--ectodomain" in message
    assert "whole-chain" in message
    assert "120 residues" in message  # names the chain length it saw


def test_ectodomain_alone_satisfies_the_topology_requirement(synthetic_inputs, tmp_path):
    result = run_pipeline(_config(synthetic_inputs, tmp_path, ectodomain=(30, 140)))
    assert result.patches or result.singletons


def test_intracellular_residues_never_reach_a_patch(synthetic_inputs, tmp_path):
    """Regression test 1: nothing below the extracellular boundary may rank."""
    result = run_pipeline(
        _config(
            synthetic_inputs,
            tmp_path,
            topology="cytoplasmic=1-67,tm=68-88,extracellular=89-120",
        )
    )
    inside = {
        r.ref_number
        for r in result.residues
        if r.topology in ("cytoplasmic", "transmembrane")
    }
    assert inside, "the fixture should have residues on the wrong side"

    ranked = {
        m.ref_number
        for patch in result.patches + result.singletons
        for m in patch.members
    }
    assert not (ranked & inside)
    # hard-excluded, not merely down-weighted
    excluded = [r for r in result.residues if not r.accessible]
    assert all(r.discrimination == 0.0 and r.composite == 0.0 for r in excluded)
    assert all("outside_topology" in r.mask_reasons for r in excluded)
    # and no sequon may be reported from there either
    assert all(
        s.ref_index is None or result.residues[s.ref_index].accessible
        for s in result.glycans.sequons
    )


def test_sequons_outside_the_extracellular_range_are_rejected_and_counted(
    synthetic_inputs, tmp_path
):
    """Regression test 1, glycan half: a cytoplasmic sequon is never occupied."""
    whole = run_pipeline(
        _config(synthetic_inputs, tmp_path / "whole", topology="whole-chain")
    )
    restricted = run_pipeline(
        _config(
            synthetic_inputs,
            tmp_path / "restricted",
            topology="cytoplasmic=1-80,extracellular=81-120",
        )
    )
    assert len(restricted.glycans.sequons) < len(whole.glycans.sequons)
    assert restricted.glycans.rejected_on_topology
    assert any(
        "never glycosylated" in w for w in restricted.glycans.warnings
    )
    # and the flags those sequons produced are gone with them
    flagged = [r for r in restricted.residues if r.glycan_flags and not r.accessible]
    assert not flagged


def test_disordered_stalk_does_not_produce_a_top_patch(tmp_path):
    """Regression test 2: a long low-pLDDT, high-divergence region is an artifact.

    The stalk is the least constrained part of a protein, so it looks
    discriminating; AlphaFold models it as an extended tether, so every residue
    looks exposed. Left in, it floods the candidate list.
    """
    inputs = synthetic.write_inputs(tmp_path / "stalk", disordered_stalk=True)

    filtered = run_pipeline(_config(inputs, tmp_path / "a", topology="whole-chain"))
    kept = run_pipeline(
        _config(inputs, tmp_path / "b", topology="whole-chain", keep_disordered=True)
    )

    assert filtered.structure.is_alphafold
    assert filtered.structure.disordered_regions, "the stalk should be detected"
    stalk = {r.ref_number for r in filtered.residues if r.in_disordered_region}
    assert len(stalk) >= 20

    in_top5 = [
        m.ref_number
        for patch in filtered.patches[:5]
        for m in patch.members
        if m.ref_number in stalk
    ]
    assert not in_top5, f"stalk residues reached the top patches: {in_top5}"
    assert all(
        "disordered_region" in r.mask_reasons
        for r in filtered.residues
        if r.in_disordered_region
    )
    # RSA inside the region is not reported as a usable number
    assert all(
        r.rsa != r.rsa for r in filtered.residues if r.in_disordered_region
    )
    # the planted epitope still comes first
    assert set(synthetic.truth_numbers()) <= {
        m.ref_number for m in filtered.patches[0].members
    }

    # and --keep-disordered genuinely overrides it, which is what makes the
    # default a choice rather than a silent truncation
    assert kept.counts["discriminating_and_exposed"] > filtered.counts[
        "discriminating_and_exposed"
    ]
    assert any(
        m.ref_number in stalk for patch in kept.patches[:5] for m in patch.members
    )
    assert any("disordered region" in w for w in filtered.warnings)
