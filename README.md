# epitope_align

Comparative epitope mapping: narrow down where a monoclonal antibody binds from
cross-species reactivity, sequence alignment and structural solvent
accessibility.

If an antibody binds the mouse and rat orthologs of a target but not the human
and marmoset ones, the epitope must be made of surface residues conserved in the
binders and changed in the non-binders. That constraint alone leaves 50-150
candidate positions on a typical ectodomain, because rodents and primates
diverged neutrally at 20-40% of positions. This tool applies the structural
filters that do the actual narrowing and returns a **ranked list of candidate
surface patches** with the experiments that would test them.

The output is a hypothesis list for chimera and point-mutant design. It is not a
prediction of the epitope, and the report says so in every run.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NanoLlama/epitope_align/blob/claude/antibody-epitope-identification-ervnyf/notebooks/epitope_mapping.ipynb)

**Not a programmer?** Start with
[GETTING_STARTED.md](GETTING_STARTED.md) — it walks through running this in a
browser via Google Colab (`notebooks/epitope_mapping.ipynb`, nothing to install)
or on your own machine, what to gather, and how to read the output.

## Install

```bash
pip install "git+https://github.com/NanoLlama/epitope_align.git@claude/antibody-epitope-identification-ervnyf"
```

or, from a clone:

```bash
pip install -e .          # Python 3.11+
pip install -e '.[dev]'   # plus pytest
```

Optional external tools, all auto-detected:

| tool | what it improves | without it |
|---|---|---|
| MAFFT (or MUSCLE) | the multiple sequence alignment | falls back to pairwise-to-reference projection, with a loud warning |
| `freesasa` | SASA calculation | Biopython's Shrake-Rupley is used instead |
| `mkdssp` | secondary structure for chimera boundaries | boundaries are geometric only, and say so |

## Quickstart

```bash
epitope-map --demo                     # worked example with a known answer
```

`--demo` writes a full set of inputs and results into `demo-run/`, and its top
patch should be exactly the eight residues it names — a one-command check that
the install is healthy. To drive the same example from files:

```bash
python examples/make_example.py        # writes a runnable synthetic dataset
epitope-map --config examples/run.yaml
```

or spelled out:

```bash
epitope-map \
  --sequences species.fasta \
  --binding binding.csv \
  --reference mouse \
  --structure AF-Q9XXXX-F1 \
  --ectodomain 25-240 \
  --outdir results/
```

`--sequences` takes a FASTA path or a list of UniProt accessions
(`--sequences mouse=Q61503,human=P21589`). `--structure` takes a PDB/mmCIF path,
a 4-character PDB ID, or an AlphaFold DB accession; the latter two are fetched
and cached. Every flag can also live in a `--config run.yaml`, and anything on
the command line overrides the file, so a run can be repeated with one tweak.

### Topology is required

An antibody reaches the outside of a cell and nothing else. Without knowing
which part of the chain faces outwards the tool would happily rank a
cytoplasmic patch first, so it **refuses to run** rather than defaulting to the
whole chain:

```
--topology extracellular=121-763,tm=68-88,cytoplasmic=1-67
--ectodomain 121-763        # treated as the extracellular range
--topology whole-chain      # a soluble protein, or an ectodomain-only construct
```

Fetch the sequences by UniProt accession and none of that is needed: the
topology is read from the entry's `TOPO_DOM` / `TRANSMEM` features, along with
its domain annotation, oligomeric state and the experimental structures that
exist for the target.

### Binding calls

```csv
species,binding
mouse,binder
rat,binder
human,non_binder
marmoset,non_binder
cyno,unknown
```

`unknown` species are aligned and shown, but excluded from scoring. The
reference must be a binder - numbering and structure are anchored to it.

## Outputs

| file | contents |
|---|---|
| `residues.tsv` | one row per reference residue: per-species residue, discrimination, Grantham severity, RSA, pLDDT, topology, domain, alignment confidence, local identity, glycosylation flag, the three composite factors kept separate, mask reasons, patch ID |
| `regions.tsv` | per-region enrichment against the whole-chain baseline: which domain, topology segment or disordered run the divergence actually sits in |
| `patches.tsv` | ranked patches with their conserved surface context and their neighbours; promoted isolated residues and singletons labelled as such |
| `chimeras.tsv` | suggested domain-swap segments, one row per segment, with a constructibility verdict |
| `mutants.tsv` | reciprocal point mutants, both directions, each numbered in its own background, marked where the residue equivalence behind them is not reliable |
| `radius_sensitivity.tsv` | which patches merge at which clustering radius (with `--radius-sweep`) |
| `patch_id_map.tsv` | previous run's patch IDs mapped onto this run's (with `--compare-run`) |
| `divergence.svg` | discrimination along the chain with domains, disordered regions and top patches marked |
| `report.md` | run parameters, alignment stats, how much signal there is, the narrowing table, top patches, glycosylation, next experiments, warnings, caveats |
| `session.pml` | PyMOL session: composite score painted white to red, top patches coloured, view set |
| `alignment.fasta` | the MSA actually used |

## How it works

1. **Alignment** (`align.py`) - MAFFT, else MUSCLE, else pairwise-to-reference.
   Per-species identity to the reference is reported and anything under ~40% is
   flagged as a possible paralog rather than an ortholog.
2. **Numbering** (`align.ResidueMap`) - one object owns every conversion between
   alignment column, reference sequence index and structure author numbering
   (insertion codes, unmodelled residues, signal-peptide offsets included). No
   other module does index arithmetic. If the reference sequence and the
   structure disagree beyond `--mismatch-tolerance`, the run stops with a diff.
3. **Discrimination** (`score.py`) - graded, not binary:
   `mean_pairwise_difference(binders, non_binders) - mean_pairwise_difference(within binders)`,
   where the pairwise difference is a normalized Grantham distance. Gaps score
   high but are flagged separately as indels; species missing at alignment edges
   are excluded rather than counted as differences.
4. **Degeneracy calibration** (`score.assess_degeneracy`) - the background rate
   is measured, not assumed: the same group sizes are re-assigned to species in
   every alternative way and the fraction of columns that still look
   discriminating is reported. A UPGMA tree over the alignment says whether the
   binding pattern simply reproduces the sequence tree, and the report leads
   with that.
5. **Structure** (`structure.py`) - SASA (freesasa or Shrake-Rupley), RSA via the
   Tien et al. (2013) maximum-ASA table, pLDDT from B-factors for AlphaFold
   models, DSSP where available, side-chain centroids with CB then CA fallback.
   Buried residues are marked and kept, never deleted.
6. **Masking** (`pipeline.py`) - residues outside the ectodomain, unmodelled,
   buried, at an assembly interface (>20% SASA lost in context) or occluded by a
   supplied partner are excluded from patch seeding, each with a recorded reason.
7. **Glycosylation** (`glycan.py`) - N-X-S/T (X != P) scanned per species on the
   ungapped sequence. Sequons that cleanly separate binders from non-binders are
   reported, and every surface residue within `--glycan-radius` of the sequon
   asparagine is flagged, because a glycan occludes at a distance.
8. **Domains** (`domains.py`) - spectral bisection of the C-alpha contact graph
   finds compact, densely self-contacting units, including discontinuous ones,
   and declines to split a single globular domain. Used for whole-domain swap
   recommendations; UniProt regions are kept as annotation only.
9. **Patches** (`patches.py`) - a graph over exposed discriminating residues with
   a 12 A cutoff (connected components, or DBSCAN with `--cluster-method dbscan`,
   which will not chain two surfaces together through one bridging residue).
   Patches are ranked raw and normalized, flagged against the 15-22 residue,
   600-900 A^2 envelope of a real conformational epitope.
10. **Experiments** (`suggest.py`) - domain-swap segments that isolate each patch,
   checked for constructibility, with a whole-domain alternative where the patch
   sits in one annotated domain; and reciprocal mutants with the gain-of-binding
   direction prioritised, since loss of binding alone can be generic misfolding.

### What the second real run changed

The v0.2.0 run on the same target kept its defensible top hit, but three things
were still wrong enough to mislead:

- **A domain swap that did not contain its patch.** The recommended construct
  held one of a patch's six residues, and it was being offered as the fallback
  whenever the segment list was declared impractical - so the tool's own advice
  pointed at the experiment guaranteed not to test its hypothesis. Domain swaps
  now require full containment, and structural domains come from a contact-graph
  decomposition (`domains.py`) rather than from UniProt's motif annotations,
  which are not the architecture anyone would swap.
- **Patch IDs assigned by rank.** Letters were reassigned between versions on
  identical input, so "patch PD" in a notebook silently meant different residues
  after a re-run. IDs are now derived from content (`p:202`), with a membership
  digest and `--compare-run` to map a previous run's IDs onto the current ones.
- **Merged surfaces that always collapsed to everything.** A transitive closure
  at any link distance groups the whole protein on a real target. Groups now
  grow by diameter and refuse the merge that would breach a footprint.

Also this round: alignment confidence is withheld unless three genuinely
different aligners ran (agreement between two settings of one program is not
evidence); ambiguity is flagged by window rather than per residue; indels get
their own geometric `indel_score` instead of being crushed by a length penalty;
the oligomer and glycan findings discount the patches they apply to instead of
only appearing in prose; and the species advisor ranks by expected value over
both binding outcomes, with real candidate orthologs listed above hypotheticals.

### What the first real run changed

Running this against mouse TfR1 produced a defensible top hit and roughly half a
report of artifact, none of it distinguishable from signal without knowing the
target. Everything below exists because of that run:

- **Topology is enforced** (above). The rank-2 patch was entirely cytoplasmic
  and carried the highest Grantham score in the run.
- **Sequons are checked against topology.** Two of three "differential
  glycosylation sites" were in the cytoplasmic tail, where N-glycosylation does
  not happen, and both had propagated proximity flags onto neighbours.
- **Disordered regions are excluded** (`structure.py`): contiguous runs of >=10
  residues averaging pLDDT < 50 are a modelling failure, not a surface. The
  stalk showed 70% of positions discriminating against a 27% baseline on the
  synthetic reproduction, purely from being unconstrained.
- **Fragmentation is visible.** Patches carry conserved surface context, their
  nearest-member distances, and a merged-surfaces section; `--radius-sweep`
  shows which fragments are one surface.
- **Isolated residues are not buried.** A high-scoring surface insertion is
  promoted rather than filed under leftovers.
- **Alignment reliability is measured** (`align.py`), and point mutants resting
  on an unreliable residue equivalence are marked UNVERIFIED. This is the
  failure that costs money: the region is right, the construct is wrong.
- **The report is reordered** so what was excluded, how much signal there is,
  the warnings and the region table all come before the rankings.
- **The enrichment figure is withheld** below six scored species instead of
  being quoted to two decimals off a handful of relabellings.

## Defaults worth knowing

| flag | default | note |
|---|---|---|
| `--rsa-cutoff` | 0.20 | buried residues are marked, not dropped |
| `--discrimination-cutoff` | 0.25 | in normalized-Grantham units, see below |
| `--patch-radius` | 12.0 A | between side-chain centroids |
| `--min-patch-size` | 2 | smaller clusters are reported as singletons |
| `--glycan-radius` | 12.0 A | how far a glycan is assumed to reach |
| `--topology` | from UniProt | required; `whole-chain` opts out explicitly |
| `--keep-disordered` | off | long low-pLDDT regions do not seed patches |
| `--radius-sweep` | off | e.g. `10,12,14,16,18` |
| `--equivalence` | `sequence` | `structural` with `--species-structure` |
| `--footprint-diameter` | 30 A | how wide a merged surface may grow |
| `--domains` | contact graph | TSV of structural domains to use instead |
| `--candidate-species` | none | orthologs to score for the panel advice |
| `--compare-run` | none | previous outdir, for `patch_id_map.tsv` |
| `--ectodomain-numbering` | `structure` | or `sequence` for 1-based reference positions |
| `--mismatch-tolerance` | 0.05 | sequence/structure disagreement before the run stops |

## Two places this departs from the original brief

**The discrimination scale.** The brief expected a perfectly discriminating
position to score "near 1.0". On a normalized Grantham scale 1.0 is the largest
chemical difference that exists (Cys<->Trp), so a clean pattern built from
ordinary substitutions lands around 0.2-0.6 and only drastic swaps approach the
top. The graded score is kept as specified - it is the chemically meaningful one
- the default cutoff is set for that scale, and a companion
`pattern_consistency` column carries the identity-based version, which does
reach 1.0 for a textbook pattern.

**Ranking.** Size normalization alone does not stop a large diffuse cluster from
outranking a tight one: a chain of weak positions running across the surface
accumulates both a high total and a high residue count. The normalized score
therefore also folds in compactness (full weight up to a 20 A spread, decaying
beyond it, since a paratope covers roughly a 20-25 A disc). The raw ranking is
reported unchanged next to it, and patches are presented in the better of the
two ranks so neither view hides a candidate.

Chimera suggestions are also emitted as several short segments per patch rather
than one range spanning it, because a conformational epitope is usually
discontinuous and swapping everything between the first and last member would
move most of the domain.

## Validation

See `validation/README.md`. In short: on a synthetic case with a planted
epitope, the top-ranked patch is exactly the planted residues (100% recall, 2x
enrichment over the naive "all exposed discriminating residues" baseline), and
the tool still reports the panel as degenerate because it is a two-clade
comparison. Adding one species that breaks the clade split cuts the candidate
set by ~30% and clears the degeneracy flag.

The real-antibody benchmark (`validation/cases/d1.3_hel.yaml`, ground truth
computed from the solved complex) has **not** been run here: this environment
blocks outbound access to RCSB and UniProt. The case file is ready to run where
those hosts are reachable.

## Development

```bash
python -m pytest tests -q            # 110 tests, all offline
python validation/benchmark.py synthetic
python validation/benchmark.py scaling
```

`tests/synthetic.py` generates the deterministic benchmark used by both the test
suite and the examples: a packed-ball structure with a buried core, insertion
codes, unmodelled residues, a numbering offset, a differential glycosylation
sequon, an indel, and a planted spatially clustered epitope.

## Worked configuration

`examples/tfr1.yaml` is the configuration the first real run should have used,
with a comment on every option explaining which artifact it prevents.

## Not implemented

From the brief's "possible extensions": KD-weighted binding calls,
multi-domain discontinuous scoring, HDX-MS/cross-linking constraints, ChimeraX
sessions and an interactive HTML report.
