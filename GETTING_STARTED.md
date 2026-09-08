# Getting started (no coding required)

This tool takes a protein that your antibody binds in some species but not in
others, and gives you back a **shortlist of surface patches** that could explain
the difference — plus the chimeras and point mutants that would test each one.

There are two ways to run it. Pick one.

---

## Route A — in your browser, using Google Colab (recommended)

Nothing to install. Colab is a free Google service that runs code on Google's
computers; you only ever press play buttons and fill in boxes.

**[▶ Open the notebook in Colab](https://colab.research.google.com/github/NanoLlama/epitope_align/blob/claude/antibody-epitope-identification-ervnyf/notebooks/epitope_mapping.ipynb)**

**Click "Copy to Drive" at the top of the page before you type anything.** A
notebook opened straight from GitHub is a read-only preview, so edits may not
stick; that button gives you your own saved copy. (`File → Save a copy in Drive`
does the same.)

Then work down the page, pressing ▶ on each grey box in order. The first one
installs the tool (about two minutes); the second runs a built-in example so you
can confirm everything works before using your own data; in the third you edit a
few lines of text between quote marks to describe your own species.

You will need a Google account, and Colab will ask you to confirm before running
a notebook it did not write — that prompt is normal.

*If the link does not open:* go to <https://colab.research.google.com>, choose
**File → Open notebook → GitHub**, paste `NanoLlama/epitope_align`, pick the
branch `claude/antibody-epitope-identification-ervnyf`,
and choose `notebooks/epitope_mapping.ipynb`.

Because Colab runs on Google's machines it has internet access, so it can fetch
sequences from UniProt and structures from AlphaFold or the Protein Data Bank
for you.

---

## Route B — on your own computer

You need Python 3.11 or newer.

- **macOS:** open **Terminal** (⌘-space, type "terminal"). Type `python3
  --version`. If it says 3.11 or higher you are set; otherwise install Python
  from <https://www.python.org/downloads/>.
- **Windows:** install Python from <https://www.python.org/downloads/>, ticking
  **"Add python.exe to PATH"** on the first screen of the installer. Then open
  **Command Prompt**.

Then, in that terminal window, one line at a time:

```bash
pip install "git+https://github.com/NanoLlama/epitope_align.git@claude/antibody-epitope-identification-ervnyf"
```

That single line is enough to get the `epitope-map` command. If you would rather
have the source too (for the examples and tests):

```bash
git clone https://github.com/NanoLlama/epitope_align.git
cd epitope_align
git checkout claude/antibody-epitope-identification-ervnyf
pip install -e .
```

Check it worked:

```bash
epitope-map --demo
```

That runs a made-up example with a known answer — six invented species, three
that bind and three that do not, with an epitope planted at residues 65, 66, 68,
70, 72, 114, 116 and 118. If the top patch it prints is exactly those, you are
ready. It also writes a full set of result files into a `demo-run` folder so you
can see what real output looks like.

**Two optional extras that improve the results.** On macOS with
[Homebrew](https://brew.sh): `brew install mafft brewsci/bio/dssp`. MAFFT gives
a proper multi-species alignment (without it the tool falls back to a cruder
method and says so loudly); DSSP lets it avoid proposing chimera boundaries that
cut a helix in half. Neither is required.

---

## What you need to gather

This is the actual work — the software is the easy part.

**1. The sequences,** one per species. The simplest way is UniProt accession
numbers: search your protein plus a species name at <https://www.uniprot.org>,
and take the code at the top of the entry (something like `P21589`). You write
them as a list with your own labels:

```
mouse=Q61503,rat=P21590,human=P21589
```

Three species is the minimum. Six to eight is far better, for a reason explained
below. If you already have the sequences in a FASTA file, you can use that
instead.

**2. Your binding results,** as a small table — one line per species, using the
same labels:

```
species,binding
mouse,binder
rat,binder
human,non_binder
marmoset,non_binder
cyno,unknown
```

`unknown` is for species you have sequences for but no binding data; they are
shown but not used for scoring. You need at least one binder and one non-binder.

In the Colab notebook this goes in a single-line box instead of a file, so the
same thing is written on one line — `mouse=binder, rat=binder, human=non_binder`.
Semicolons work as well as commas, `yes`/`no` are accepted, and the notebook
prints back what it understood before anything runs.

**3. A reference species.** One of the binders — everything in the results is
numbered according to it.

**4. Which part of the protein is outside the cell.** An antibody can only reach
the outside, so the tool will not run without knowing where that is - it stops
and asks rather than ranking residues inside the cell. If you gave UniProt
accessions above, leave it alone: it is read from the UniProt entry
automatically. Otherwise pass `--topology extracellular=121-763,tm=68-88,cytoplasmic=1-67`
(the numbers come from the "Subcellular location" / topology section of the
UniProt page), or `--topology whole-chain` if your protein is soluble or your
sequences are already trimmed to the ectodomain.

**5. A 3D structure of that reference species.** Any of:

- an AlphaFold model, written as `AF-` plus the reference species' UniProt
  accession plus `-F1` — for example `AF-Q61503-F1`. Every UniProt entry links
  its AlphaFold model, and almost every protein has one.
- a Protein Data Bank ID like `4H2I`, if someone has solved the structure
  experimentally
- your own `.pdb` or `.cif` file

**6. Optionally, the region you care about** — for a cell-surface receptor,
usually the part outside the cell. Written as `25-240`, in the structure's own
numbering. Leave it out to analyse the whole thing.

Then the whole run is one line:

```bash
epitope-map \
  --sequences mouse=Q61503,rat=P21590,human=P21589 \
  --binding binding.csv \
  --reference mouse \
  --structure AF-Q61503-F1 \
  --ectodomain 25-240 \
  --outdir results
```

---

## Reading the results

Open **`results/report.md`** first — it is written to be read start to finish,
and deliberately ordered so the things that would undermine a hit come before
the hit. Five parts matter most:

**"What was excluded, and why"** lists the parts of the protein an antibody
cannot reach, and any region the structure prediction did not really model. If
this says the whole chain was treated as accessible and your protein sits in a
membrane, stop and set the topology.

**"How much signal is there?"** Read this before anything else. If your binders
are all rodents and your non-binders are all primates, then *every* position
where rodents and primates happen to differ looks meaningful, and there will be
50–150 of them. The report says so plainly and tells you how much of what you
are seeing is explainable by chance. A run flagged this way is not wrong, but
its shortlist is weak.

**"Where the divergence actually sits"** breaks the chain into regions and shows
the percentage of discriminating positions in each against the whole-chain
baseline. A region far above the baseline is either your answer or an artifact —
the least constrained parts of a protein (stalks, linkers) look discriminating
everywhere without being epitopes anywhere. `divergence.svg` is the same thing
as a picture.

**"Narrowing"** shows how many candidate residues survive each filter, so you can
see where the shortlist actually came from.

**"Top candidate patches"** is the answer: groups of residues that sit together
on the surface, are conserved in the species that bind and changed in the ones
that do not. Each lists its members, how exposed they are, and how chemically
drastic the substitutions are.

**"Suggested next experiments"** turns each patch into lab work: which segments
to swap between species, and which point mutants to make. Prioritise the
**gain-of-binding** mutants — putting a binder's residue into a species that
does not bind. If that restores binding, it is hard to argue with. Loss of
binding on its own can just mean you broke the protein.

Watch for two labels here. A mutant marked **UNVERIFIED** sits in a stretch
where the sequence alignment is ambiguous: the region is worth testing but the
specific residue may be the wrong one, so swap the segment before ordering the
point mutant. A chimera marked **NOT CONSTRUCTIBLE** needs too many pieces or
crosses the membrane; use the whole-domain swap offered alongside it.

The other files: `patches.tsv` and `residues.tsv` are spreadsheets (open them in
Excel — they are tab-separated) with every number behind the report;
`mutants.tsv` and `chimeras.tsv` are the experiment lists; `session.pml` opens
your structure in PyMOL with the candidate patches coloured in.

---

## Things worth knowing before you trust it

- **It is a shortlist, not an answer.** The report says this in several places
  and means it. The output is there to make your next ten experiments smarter,
  not to replace them.
- **More species is the single biggest improvement available.** Especially a
  species that breaks the pattern — a rodent that does *not* bind, or a primate
  that does. Each informative species roughly halves the candidate list, which
  no amount of cleverness in the software can match.
- **The cause need not be inside the footprint.** A change just outside the
  contact area can move a loop; a sugar attached nearby can block the antibody
  from several ångströms away. The tool flags both situations rather than hiding
  them.
- **Warnings are information, not failure.** Every run prints the specific
  limitations of that run. They are worth reading.

---

## If something goes wrong

**"No module named 'epitope_map.notebook'"** (or any other missing part of the
package) — the Colab runtime is holding an older copy of the tool than the
notebook expects. Re-run **Step 1**; it now forces a fresh install. If it still
complains, choose **Runtime → Restart session** and run Step 1 again.

**"command not found: epitope-map"** — the install did not finish, or a new
terminal window lost it. Re-run `pip install -e .` from inside the project
folder.

**"the membrane topology of the reference could not be determined"** — this is
the tool refusing to score the inside of a cell. Give it `--topology` as
described above, or `--topology whole-chain` for a soluble protein.

**"reference species ... is marked non_binder"** — the reference has to be a
species the antibody binds.

**"binding call for X has no matching sequence"** — a label in your binding table
does not match any sequence label. Check the spelling.

**"reference sequence and structure disagree at ..."** — the structure is not of
the species you named as reference, or it is the right protein but the wrong
chain. Try adding `--chain A` (or B, C…). The message lists the mismatches so
you can see what happened.

**Anything about "unable to fetch"** — the computer running the tool cannot
reach UniProt or the PDB. In Colab this should not happen; on a work laptop
behind a firewall it might. Download the files by hand and point at them
instead.
