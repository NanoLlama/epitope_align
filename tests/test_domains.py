"""Structural domain decomposition from the contact graph."""

import pytest

from epitope_map import demo
from epitope_map.domains import StructuralDomain, decompose, describe
from epitope_map.structure import load_structure


def _decompose(model):
    keys = {record.key: index for index, record in enumerate(model.residues)}
    return decompose(model, lambda key: keys.get(key))


def test_two_lobes_are_found_as_two_domains(tmp_path):
    model = load_structure(str(demo.write_two_lobe_pdb(tmp_path / "two.pdb")))
    domains = _decompose(model)
    assert len(domains) == 2
    assert all(d.size >= 40 for d in domains)
    # and the split is where the linker is, not somewhere arbitrary
    first, second = sorted(domains, key=lambda d: min(d.ref_indices))
    assert max(first.ref_indices) < min(second.ref_indices)


def test_a_single_globule_is_not_split(tmp_path):
    """Forcing a split on a compact protein would invent boundaries."""
    paths = demo.write_inputs(tmp_path / "one")
    model = load_structure(str(paths["structure"]))
    domains = _decompose(model)
    assert len(domains) == 1
    assert domains[0].name == "whole chain"


def test_domains_may_be_discontinuous():
    """A real domain is often two stretches - the decomposition must allow it."""
    domain = StructuralDomain(name="D1", ref_indices=[1, 2, 3, 40, 41, 42])
    assert domain.ranges() == [(1, 3), (40, 42)]
    assert domain.size == 6


def test_describe_uses_reference_numbering(tmp_path):
    paths = demo.write_inputs(tmp_path / "one")
    model = load_structure(str(paths["structure"]))

    class FakeMap:
        def number_of(self, index):
            return str(index + 25)

    domain = StructuralDomain(name="D1", ref_indices=[0, 1, 2, 10, 11])
    assert describe(domain, FakeMap()) == "25-27+35-36"


def test_pipeline_uses_structural_domains_not_uniprot_regions(tmp_path):
    """UniProt regions are annotation; the swap tier must not use them."""
    from epitope_map.pipeline import RunConfig, run_pipeline

    paths = demo.write_inputs(tmp_path / "in")
    result = run_pipeline(
        RunConfig(
            sequences=str(paths["sequences"]),
            binding=str(paths["binding"]),
            reference="mouse",
            structure=str(paths["structure"]),
            topology="whole-chain",
            outdir=tmp_path / "out",
        )
    )
    assert result.structural_domains
    assert "contact graph" in result.domain_source
    assert all(r.domain_source in ("", result.domain_source) for r in result.residues)


def test_user_supplied_domains_win(tmp_path):
    from epitope_map.pipeline import RunConfig, run_pipeline

    paths = demo.write_inputs(tmp_path / "in")
    table = tmp_path / "domains.tsv"
    table.write_text("name\tstart\tend\nlobe_one\t1\t60\nlobe_two\t61\t120\n")
    result = run_pipeline(
        RunConfig(
            sequences=str(paths["sequences"]),
            binding=str(paths["binding"]),
            reference="mouse",
            structure=str(paths["structure"]),
            topology="whole-chain",
            domains=str(table),
            outdir=tmp_path / "out",
        )
    )
    assert result.domain_source.startswith("user table")
    assert {d.name for d in result.structural_domains} == {"lobe_one", "lobe_two"}
    assert result.residues[0].domain == "lobe_one"
    assert result.residues[-1].domain == "lobe_two"


def test_a_user_domain_may_be_discontinuous(tmp_path):
    """TfR1's protease-like domain is 121-188 plus 384-606, and is one domain."""
    from epitope_map.pipeline import RunConfig, run_pipeline

    paths = demo.write_inputs(tmp_path / "in")
    table = tmp_path / "domains.tsv"
    table.write_text(
        "name\tstart\tend\n"
        "protease-like\t1\t30\n"
        "apical\t31\t70\n"
        "protease-like\t71\t120\n"
    )
    result = run_pipeline(
        RunConfig(
            sequences=str(paths["sequences"]),
            binding=str(paths["binding"]),
            reference="mouse",
            structure=str(paths["structure"]),
            topology="whole-chain",
            domains=str(table),
            outdir=tmp_path / "out",
        )
    )
    by_name = {d.name: d for d in result.structural_domains}
    assert set(by_name) == {"protease-like", "apical"}
    protease = by_name["protease-like"]
    assert len(protease.ranges()) == 2  # one domain, two stretches
    assert protease.size == 80

    # a patch inside one stretch is contained by the whole domain
    assert result.residues[5].domain == "protease-like"
    assert result.residues[100].domain == "protease-like"
    assert result.residues[50].domain == "apical"
