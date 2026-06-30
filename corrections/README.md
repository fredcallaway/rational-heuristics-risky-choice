# Correction workflow

This directory produces the marked-up **`corrections.pdf`** for the published paper. All
the code in this directory was written by GPT 5.5 and Opus 4.8 and audited by the second
author (Fred Callaway).

Many of the errors resulted from confusion about what "dispersion" means. The definition
we ultimately adopt follows Payne et al. (1988). High dispersion means that the outcome
probabilities are very different from one another; one outcome can be much more likely
than all others. This is the opposite of the usual statistical definition of dispersion,
where maximal dispersion corresponds to all outcomes being equally likely. Parts of the
code assumed the standard statistical definition. As a result, all line plots showing the
effect of dispersion were flipped along the x axis. This also affected two places in the
text where an effect was described as being opposite the true direction:

1. Higher dispersion in fact leads to more *attribute-based* processing (original:
   alternative-based).
2. Higher dispersion in fact *increases* decision qualtiy (original: decreases).

A second class of errors arised from incorrectly parsing lme4 output. Intercepts were
reported as regressions. Fortunately, these errors only affected the reported statistical results.
The textual descriptions were and are correct. Shockingly, reported p values (all p<.001) remain
the same; this is explained by the large sample size and a healthy dose of dumb luck.

The final class of error is of the classic copy-paste variety. As with the second class, this
affected the reported statistics only, not the textual description. In one case, a p=.015 result
was reported as p<.001. Again, we were extremely lucky here.

*end human-generated content*

## Run it

From this directory (corrections/) run:

```bash
./run_corrections.sh
```

This:

1. **Re-derives** the Experiment 1 regression coefficients in R from the processed
   `code/data/human/1.0/processed/trials.csv` (no clustering, no main pipeline; ~1 min).
2. **Verifies** those coefficients match the corrected B-values baked into the PDF
   generator. **The run aborts if any value disagrees** — the PDF is not built on failure.
3. **Builds** `corrections.pdf` and `corrections.log` at the repo root.

Requirements: `Rscript` with the `lme4` package, and the user's default Python with
`pymupdf` installed. Override the interpreters with the `RSCRIPT` / `PYTHON` env vars.

## Outputs

| File | What it is |
| --- | --- |
| `<repo>/corrections.pdf` | `published.pdf` marked up: strikethroughs over old text with the corrected text in the comment pane; margin notes for the four figure replacements. |
| `<repo>/corrections.log` | Plain-text log of every annotation (`old` → `new`), tagged by kind. |
| `corrections/output/{strategy,processing}_regression_results.txt` | The R re-derivation, with each coefficient labelled by the prose it backs. |
| `corrections/output/verification.log` | Per-coefficient PASS/FAIL of the PDF-vs-R check. |

## The consistency proof

`regression_corrections.py` is the **single source of truth** for the corrected B-values in
the two regression sections (Exp1 strategy logistic regressions, journal p. 913; Exp1
processing-pattern mixed model, p. 915). Both the PDF generator and the verifier import it,
so a number cannot drift between "what the PDF claims" and "what is checked".

`verify_stats.py` parses the `Estimate=…` from each R output block, rounds it to the
precision shown in the correction, and asserts it equals the value in
`regression_corrections.py`. Each record's `key` (e.g. `SAT_TTB ~ sigma`) is the join key
with the `[… ~ …]` tag in the R output.

### What is and isn't proven

- **Proven against R:** the 8 Experiment 1 regression coefficients (7 strategy + 1
  processing-pattern). These are the only *updated numerical* results in the correction.
- **Not re-derived here** (documented in the log as `chi2` / `kappa`):
  - The seven Experiment 2 chi-square / effect-size values on p. 919 are copy-paste fixes.
    They are corrected in the PDF but not independently recomputed by this workflow.
  - The Figure B1 participant kappa on p. 931 is also a file-reference fix.
  - The remaining changes are wording fixes following from the inverted dispersion
    interpretation, plus typos; they have no numerical result to verify.

## Regenerating the replacement figures

The four replacement figures in `<repo>/new-figs/` are the **pre-approved** versions sent
by a co-author, and the workflow above does not touch them. `make_new_figs.py` re-derives
them self-contained (a trimmed copy of `code/python/make_figures.py::exp1_condition_lines`
with the inverse-alpha flip), reading only the already-processed model/human trial CSVs:

```bash
python make_new_figs.py            # regenerate to a temp dir, leaving new-figs/ untouched
python make_new_figs.py --write    # overwrite new-figs/
```

It defaults to a temp dir because the bitmaps are not byte-identical across matplotlib /
font versions (and the error bars use a bootstrap, so they vary slightly per run); the
regenerated figures match the approved ones structurally. Keep the committed, approved
`new-figs/` for the submission unless you have a reason to replace them.

## Files

| File | Role |
| --- | --- |
| `run_corrections.sh` | Single entry point (R → verify → PDF). |
| `regression_corrections.py` | Shared source of truth for the corrected regression B-values. |
| `verify_stats.py` | Checks the PDF's regression values against the R output; exits non-zero on mismatch. |
| `make_corrections_pdf.py` | Marks up `published.pdf` → `corrections.pdf`, writes `corrections.log`. |
| `make_new_figs.py` | Optional self-contained regen of the four `new-figs/` replacement figures. |
| `../R/strategy_regression_standalone.R` | Re-derives the Exp1 strategy logistic regressions. |
| `../R/processing_regression_standalone.R` | Re-derives the Exp1 processing-pattern mixed model. |
| `output/` | Generated R results and verification log. |

The full list of correction changes (old/new text, keyed to journal page numbers) lives in
`<repo>/CHANGES.md`.
