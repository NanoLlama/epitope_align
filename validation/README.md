# Validation

Run with:

```bash
python validation/benchmark.py synthetic          # planted-epitope positive control
python validation/benchmark.py scaling            # what one more species buys
python validation/benchmark.py validation/cases/d1.3_hel.yaml   # needs network
```

## What has actually been run here

### 1. Synthetic positive control (offline, part of the test suite)

`tests/synthetic.py` builds a 120-residue packed-ball structure with a planted,
spatially clustered 8-residue epitope, three "rodent" binders and three
"primate" non-binders. Neutral clade markers are scattered over the rest of the
surface at the same rate a real two-clade comparison produces them; the planted
substitutions are chemically drastic, the neutral ones conservative.

```
case: synthetic (planted epitope)
  true contact residues:          8 (65, 66, 68, 70, 72, 114, 116, 118)
  reference residues analysed:    120
  naive baseline (exposed+disc.): 16
  top patch size:                 8
  true epitope hit in top 1 / 3:  True / True
  recall in top 1 / top 3:        100% / 100%
  candidates tested to 1st hit:   1
  enrichment over baseline:       2.00x
  flagged as degenerate:          True
```

The last line matters as much as the recall: the panel is two clades, and the
tool says so even when it happens to get the right answer.

### 2. Degenerate case, deliberately

`tests/test_pipeline.py::test_two_clade_run_reports_weak_discrimination` and
`tests/test_score.py::test_two_clade_comparison_is_reported_as_degenerate` assert
that a clean rodent-vs-primate split is reported as degenerate, with the
background discriminating rate computed from all alternative labellings of the
same species rather than assumed.

### 3. Species scaling

```
panel                   species  candidates  top_patch  recall_top1  clade_split
two clades              6        16          8          1.0          True
+1 informative species  7        11          8          1.0          False
```

Adding one rodent that carries the epitope substitutions on a rodent background
cuts the candidate set by ~30% and removes the clade degeneracy entirely, which
is the concrete version of the report's standing advice to add species.

## What has *not* been run here

The real-antibody case (`cases/d1.3_hel.yaml`) has **not** been executed in this
environment: outbound access to `files.rcsb.org` and `rest.uniprot.org` is
refused by the sandbox egress policy (403 on CONNECT), so neither the structure
nor the ortholog sequences could be fetched. The case file is complete and will
run where those hosts are reachable; its ground truth is derived from the 1VFB
complex by the harness itself, so it does not depend on a hand-copied epitope
list.

Its binding table ships with two species only, which is not a validated
cross-species panel - see the comments in the YAML before using it as one.

## What the first real run added

Running v0.1.0 against mouse TfR1 (Q62351) vs rat / human / marmoset on
`AF-Q62351-F1` produced a defensible top hit and a large amount of artifact
alongside it. The regression tests below now cover each failure:

| failure on the real target | test |
|---|---|
| rank-2 patch entirely cytoplasmic | `test_intracellular_residues_never_reach_a_patch` |
| two of three "differential sequons" in the cytoplasmic tail | `test_sequons_outside_the_extracellular_range_are_rejected_and_counted` |
| six patches from a 32-residue disordered stalk | `test_disordered_stalk_does_not_produce_a_top_patch` |
| one apical surface split across four entries | `test_groups_20A_apart_separate_at_12A_and_merge_at_18A`, `test_min_distance_beats_centroid_separation_for_elongated_patches` |
| highest-scoring residue in the run filed as a low-priority singleton | `test_isolated_indel_is_promoted_not_buried` |
| point mutants named from an ambiguous alignment window | `test_confidence_is_lower_in_the_ambiguous_window`, `test_unverified_mutants_are_marked_and_deprioritised` |
| offset shifting mid-patch with nothing verifying it | `test_equivalences_round_trip_across_an_indel` |
| enrichment quoted to two decimals off six labellings | `test_enrichment_is_withheld_for_a_small_panel` |

`examples/tfr1.yaml` is the configuration that run should have used. It needs
network access, so it has not been executed here either.

## What the second real run added

The v0.2.0 run on the same target kept its top hit but still produced three
misleading things. Each now has a test:

| failure on the real target | test |
|---|---|
| domain swap containing 1 of a patch's 6 residues, offered as the fallback for a non-constructible segment list | `test_domain_swap_requires_the_domain_to_contain_the_whole_patch`, `test_impractical_segments_are_not_redirected_to_a_wrong_domain_swap` |
| patch letters reassigned between versions on identical input | `test_patch_ids_are_derived_from_content_not_rank`, `test_changing_one_patch_leaves_the_others_identified_the_same` |
| merged surfaces collapsing to one 83 A group every time | `test_merging_grows_by_diameter_not_by_chaining`, `test_a_merge_that_would_breach_the_footprint_is_refused` |
| combined area double-counting overlapping patches | `test_merged_area_is_the_union_not_a_sum` |
| `alignment_confidence` 1.0 everywhere, so identity did all the work unannounced | `test_confidence_is_blank_rather_than_1_when_nothing_could_disagree` |
| ambiguity flagged per residue either side of a cutoff | `test_window_marks_every_residue_in_it_uniformly` |
| a real surface insertion crushed from 0.76 to 0.25 | `test_a_single_residue_surface_insertion_is_not_crushed`, `test_indel_in_a_buried_helix_scores_below_one_in_an_exposed_coil` |
| advisor ranking an outcome nobody can order | `test_a_real_ortholog_outranks_an_outcome_chosen_hypothetical` |
| oligomer and glycan warnings never touching the score | `test_a_wholly_glycan_proximal_patch_is_discounted_and_labelled`, `test_penalties_change_the_ranking_not_just_the_prose` |
| "within the analysed range" contradicting the topology exclusion above it | `test_baseline_counts_track_each_narrowing_step` |

## Metrics reported

* **hit in top 1 / top 3** - does the top-ranked (or any of the top three)
  patch contain a true contact residue?
* **recall** - fraction of true contact residues inside those patches.
* **candidates tested to first hit** - walking the ranked patch list residue by
  residue, how many must be tested before the first true contact is reached.
* **enrichment over baseline** - density of true contacts in the top patch
  divided by their density in the naive "all exposed discriminating residues"
  set. 1.0x means the ranking added nothing.
