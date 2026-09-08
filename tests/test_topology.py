"""Topology parsing, domain annotation and the refusal to guess."""

import pytest

from epitope_map.io_seq import InputError
from epitope_map.topology import (
    BOUNDARY_MARGIN,
    Topology,
    TopologyUnknown,
    domain_at,
    load_domains_tsv,
    parse_topology_spec,
    parse_uniprot_features,
    require_topology,
    topology_from_range,
)

SPEC = "extracellular=121-763,tm=68-88,cytoplasmic=1-67"


def test_spec_parsing_and_lookup():
    topology = parse_topology_spec(SPEC)
    assert topology.kind_at(30) == "cytoplasmic"
    assert topology.kind_at(70) == "transmembrane"
    assert topology.kind_at(500) == "extracellular"
    assert topology.kind_at(100) == "unknown"       # in the gap between segments
    assert topology.is_accessible(500)
    assert not topology.is_accessible(30)
    assert not topology.is_accessible(70)
    assert not topology.is_accessible(100)


@pytest.mark.parametrize(
    "spec",
    [
        "extracellular=121-763",
        "extracellular:121-763",
        "ectodomain=121-763",
        "extra = 121-763",
        "extracellular=121..763",
    ],
)
def test_spec_accepts_the_obvious_spellings(spec):
    assert parse_topology_spec(spec).extracellular_ranges() == [(121, 763)]


def test_multiple_ranges_for_one_label():
    topology = parse_topology_spec("extracellular=20-40|60-80")
    assert topology.extracellular_ranges() == [(20, 40), (60, 80)]
    assert topology.is_accessible(30) and topology.is_accessible(70)
    assert not topology.is_accessible(50)


def test_whole_chain_is_the_explicit_opt_out():
    topology = parse_topology_spec("whole-chain")
    assert topology.whole_chain
    assert topology.is_accessible(1) and topology.is_accessible(10_000)
    assert "declared by the user" in topology.summary()


def test_bad_specs_are_rejected_with_advice():
    with pytest.raises(InputError, match="expected"):
        parse_topology_spec("121-763")
    with pytest.raises(InputError, match="unknown topology label"):
        parse_topology_spec("nucleus=1-50")
    with pytest.raises(InputError, match="start after end"):
        parse_topology_spec("extracellular=763-121")
    with pytest.raises(InputError):
        parse_topology_spec("")


def test_boundary_proximity():
    topology = parse_topology_spec("extracellular=100-200")
    assert topology.near_boundary(101, margin=BOUNDARY_MARGIN)
    assert topology.near_boundary(199, margin=BOUNDARY_MARGIN)
    assert not topology.near_boundary(150, margin=BOUNDARY_MARGIN)


def test_uniprot_features_give_topology_and_domains():
    payload = {
        "sequence": {"length": 763},
        "features": [
            {"type": "Topological domain", "location": {"start": {"value": 1}, "end": {"value": 67}}, "description": "Cytoplasmic"},
            {"type": "Transmembrane", "location": {"start": {"value": 68}, "end": {"value": 88}}, "description": "Helical"},
            {"type": "Topological domain", "location": {"start": {"value": 89}, "end": {"value": 763}}, "description": "Extracellular"},
            {"type": "Domain", "location": {"start": {"value": 190}, "end": {"value": 380}}, "description": "Apical"},
            {"type": "Binding site", "location": {"start": {"value": 200}, "end": {"value": 200}}, "description": "ignored"},
        ],
    }
    topology, domains = parse_uniprot_features(payload)
    assert topology.length == 763
    assert topology.source == "UniProt features"
    assert not topology.is_accessible(50)
    assert topology.is_accessible(200)
    assert [d.description for d in domains] == ["Apical"]
    assert domain_at(domains, 200).description == "Apical"
    assert domain_at(domains, 500) is None


def test_uniprot_entry_without_topology_stays_unknown():
    topology, domains = parse_uniprot_features({"sequence": {"length": 100}, "features": []})
    assert not topology.known
    assert domains == []
    with pytest.raises(TopologyUnknown):
        require_topology(topology, 100, "mouse")


def test_require_topology_names_every_way_out():
    with pytest.raises(TopologyUnknown) as excinfo:
        require_topology(None, 763, "mouse")
    message = str(excinfo.value)
    assert "--topology" in message
    assert "--ectodomain" in message
    assert "whole-chain" in message
    assert "UniProt" in message
    assert "763" in message


def test_require_topology_passes_when_known():
    topology = parse_topology_spec(SPEC)
    assert require_topology(topology, 763, "mouse") is topology
    whole = parse_topology_spec("whole-chain")
    assert require_topology(whole, 763, "mouse") is whole


def test_range_becomes_the_extracellular_segment():
    topology = topology_from_range(25, 240)
    assert topology.is_accessible(100)
    assert not topology.is_accessible(10)


def test_domains_tsv(tmp_path):
    path = tmp_path / "domains.tsv"
    path.write_text("name\tstart\tend\nApical\t190\t380\nHelical\t381\t600\n")
    domains = load_domains_tsv(path)
    assert [d.description for d in domains] == ["Apical", "Helical"]
    with pytest.raises(InputError):
        (tmp_path / "bad.tsv").write_text("Apical\t190\n")
        load_domains_tsv(tmp_path / "bad.tsv")


def test_unknown_topology_is_permissive_only_when_nothing_is_known():
    """An empty topology must not silently mask everything out."""
    assert Topology().is_accessible(1)
