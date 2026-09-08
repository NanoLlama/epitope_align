"""Structure parsing, SASA/RSA, pLDDT and context masking."""

from pathlib import Path

import pytest

from epitope_map import demo as synthetic
from epitope_map.structure import ResidueKey, StructureError, load_structure

ATOM = (
    "ATOM  {serial:5d}  {name:<3s}{resname:>4s} {chain}{resseq:4d}{icode}"
    "   {x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}          {element:>2s}"
)


def write_pdb(path: Path, residues, b=85.0):
    """residues: list of (chain, resseq, icode, resname, (x, y, z))."""
    lines = []
    serial = 1
    for chain, resseq, icode, resname, (x, y, z) in residues:
        atoms = [("N", -1.2), ("CA", 0.0), ("C", 1.2), ("O", 1.6)]
        if resname != "GLY":
            atoms.append(("CB", 0.8))
        for name, dx in atoms:
            lines.append(
                ATOM.format(
                    serial=serial, name=name, resname=resname, chain=chain,
                    resseq=resseq, icode=icode, x=x + dx, y=y + (0.9 if name == "CB" else 0.0),
                    z=z, b=b, element=name[0],
                )
            )
            serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_first_protein_chain_is_used_by_default(tmp_path):
    residues = [("A", i, " ", "ALA", (i * 4.0, 0.0, 0.0)) for i in range(1, 30)]
    residues += [("B", i, " ", "GLY", (i * 4.0, 20.0, 0.0)) for i in range(1, 30)]
    model = load_structure(str(write_pdb(tmp_path / "two.pdb", residues)))
    assert model.chain_id == "A"
    assert len(model.residues) == 29


def test_named_chain_and_missing_chain_error(tmp_path):
    residues = [("A", i, " ", "ALA", (i * 4.0, 0.0, 0.0)) for i in range(1, 30)]
    residues += [("B", i, " ", "GLY", (i * 4.0, 20.0, 0.0)) for i in range(1, 30)]
    path = write_pdb(tmp_path / "two.pdb", residues)
    assert load_structure(str(path), chain_id="B").chain_id == "B"
    with pytest.raises(StructureError, match="chain"):
        load_structure(str(path), chain_id="Z")


def test_rsa_distinguishes_buried_from_exposed(tmp_path):
    paths = synthetic.write_inputs(tmp_path)
    model = load_structure(str(paths["structure"]))
    core = [model.residues[i].rsa for i in synthetic.CORE_INDICES]
    outer = [
        model.residues[i].rsa
        for i in synthetic.EPITOPE_INDICES + synthetic.NOISE_INDICES
        if i < min(synthetic.UNMODELLED_INDICES)
    ]
    assert max(core) < 0.20  # every packed-core residue is below the cutoff
    assert min(outer) > 0.20  # every outer-layer residue is above it
    assert sum(core) / len(core) < 0.1 * (sum(outer) / len(outer))


def test_glycine_falls_back_to_ca_centroid(tmp_path):
    residues = [("A", 1, " ", "GLY", (0.0, 0.0, 0.0))]
    residues += [("A", i, " ", "ALA", (i * 4.0, 0.0, 0.0)) for i in range(2, 25)]
    model = load_structure(str(write_pdb(tmp_path / "gly.pdb", residues)))
    glycine = model.residues[0]
    assert glycine.aa == "G"
    assert glycine.centroid_source in ("CB", "CA")
    assert glycine.centroid is not None


def test_plddt_read_from_b_factors_only_for_alphafold_models(tmp_path):
    residues = [("A", i, " ", "ALA", (i * 4.0, 0.0, 0.0)) for i in range(1, 30)]
    plain = load_structure(str(write_pdb(tmp_path / "plain.pdb", residues, b=45.0)))
    assert plain.is_alphafold is False
    assert plain.residues[0].plddt != plain.residues[0].plddt  # NaN

    af = load_structure(str(write_pdb(tmp_path / "AF-Q9TEST-F1.pdb", residues, b=45.0)))
    assert af.is_alphafold is True
    assert af.residues[0].plddt == pytest.approx(45.0)


def test_out_of_range_b_factors_are_not_treated_as_plddt(tmp_path):
    residues = [("A", i, " ", "ALA", (i * 4.0, 0.0, 0.0)) for i in range(1, 30)]
    model = load_structure(str(write_pdb(tmp_path / "AF-Q9TEST-F1.pdb", residues, b=180.0)))
    assert model.is_alphafold is False
    assert any("B-factors" in w for w in model.warnings)


def test_assembly_context_flags_interface_residues(tmp_path):
    residues = [("A", i, " ", "ALA", (i * 4.0, 0.0, 0.0)) for i in range(1, 25)]
    # chain B sits right on top of the first few A residues
    residues += [("B", i, " ", "ALA", (i * 4.0, 4.5, 0.0)) for i in range(1, 5)]
    path = write_pdb(tmp_path / "complex.pdb", residues)
    model = load_structure(str(path), chain_id="A", context_chains=["B"])
    interface = [r.key.resseq for r in model.residues if r.buried_by_context]
    assert interface, "chain B should occlude part of chain A"
    assert all(number <= 6 for number in interface)
    assert not model.residues[-1].buried_by_context


def test_insertion_codes_and_numbering_gaps_are_preserved(tmp_path):
    paths = synthetic.write_inputs(tmp_path)
    model = load_structure(str(paths["structure"]))
    labels = [r.key.label for r in model.residues]
    assert any(label[-1].isalpha() for label in labels)
    assert any("internal residue-numbering gap" in w for w in model.warnings)


def test_residue_key_label_formatting():
    assert ResidueKey("A", 45).label == "45"
    assert ResidueKey("A", 45, "B").label == "45B"
