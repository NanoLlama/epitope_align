"""Oligomeric state and experimental templates, read from the UniProt entry."""

from epitope_map.topology import assembly_evidence, assembly_warnings

PAYLOAD = {
    "comments": [
        {
            "commentType": "SUBUNIT",
            "texts": [{"value": "Homodimer; disulfide-linked. Binds transferrin."}],
        }
    ],
    "uniProtKBCrossReferences": [
        {
            "database": "PDB",
            "id": "1CX8",
            "properties": [
                {"key": "Method", "value": "X-ray"},
                {"key": "Resolution", "value": "3.20 A"},
                {"key": "Chains", "value": "A/B=121-760"},
            ],
        },
        {
            "database": "PDB",
            "id": "6H5I",
            "properties": [
                {"key": "Method", "value": "EM"},
                {"key": "Resolution", "value": "2.80 A"},
            ],
        },
        {"database": "AlphaFoldDB", "id": "Q62351"},
    ],
}


def test_oligomeric_state_and_templates_are_read():
    evidence = assembly_evidence(PAYLOAD)
    assert evidence.oligomeric
    assert "Homodimer" in evidence.subunit_text
    assert [t["pdb_id"] for t in evidence.templates] == ["1CX8", "6H5I"]
    assert evidence.templates[0]["chains"] == "A/B=121-760"
    assert evidence.best_template["pdb_id"] == "6H5I"  # best resolution wins


def test_monomer_prediction_of_an_oligomer_is_warned_about():
    evidence = assembly_evidence(PAYLOAD)
    messages = assembly_warnings(evidence, is_alphafold=True, context_supplied=False)
    assert any("predicted" in m and "solvent-exposed" in m for m in messages)
    assert any("--assembly-context" in m for m in messages)


def test_supplying_the_assembly_silences_the_warning():
    evidence = assembly_evidence(PAYLOAD)
    messages = assembly_warnings(evidence, is_alphafold=True, context_supplied=True)
    assert not any("solvent-exposed" in m for m in messages)


def test_experimental_structures_are_offered_over_a_prediction():
    evidence = assembly_evidence(PAYLOAD)
    messages = assembly_warnings(evidence, is_alphafold=True, context_supplied=True)
    assert any("--structure 6H5I" in m for m in messages)


def test_a_monomeric_protein_produces_no_assembly_warning():
    payload = {
        "comments": [{"commentType": "SUBUNIT", "texts": [{"value": "Monomer."}]}],
        "uniProtKBCrossReferences": [],
    }
    evidence = assembly_evidence(payload)
    assert not evidence.oligomeric
    assert assembly_warnings(evidence, is_alphafold=True, context_supplied=False) == []


def test_an_entry_without_comments_is_handled():
    evidence = assembly_evidence({})
    assert not evidence.oligomeric
    assert evidence.templates == []
    assert evidence.best_template is None
