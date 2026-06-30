#!/usr/bin/env bash
# Reproducible correction workflow for "Identifying Resource-Rational Heuristics for
# Risky Choice". Single command:
#
#     code/corrections/run_corrections.sh
#
# Steps:
#   1. Run the two standalone R scripts to RE-DERIVE the corrected Exp1 regression
#      coefficients from the processed trials.csv (no clustering, no main pipeline).
#   2. VERIFY those coefficients match the corrected B-values baked into the PDF
#      generator (regression_corrections.py). Fails the whole run on any mismatch.
#   3. Build corrections.pdf (published.pdf marked up) and corrections.log.
#
# Inputs (must exist):  ../data/human/1.0/processed/trials.csv,  <repo>/published.pdf
# Outputs:  <repo>/corrections.pdf, <repo>/corrections.log,
#           corrections/output/{strategy,processing}_regression_results.txt,
#           corrections/output/verification.log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/.." && pwd)"
OUT="$HERE/output"
mkdir -p "$OUT"

PYTHON="${PYTHON:-python}"
RSCRIPT="${RSCRIPT:-Rscript}"
TRIALS="$CODE/data/human/1.0/processed/trials.csv"

STRAT_OUT="$OUT/strategy_regression_results.txt"
PROC_OUT="$OUT/processing_regression_results.txt"

echo "==> [1/3] Re-deriving Exp1 regression coefficients in R (this takes ~1 min)"
"$RSCRIPT" --vanilla "$CODE/R/strategy_regression_standalone.R"   "$TRIALS" "$STRAT_OUT"
"$RSCRIPT" --vanilla "$CODE/R/processing_regression_standalone.R" "$TRIALS" "$PROC_OUT"

echo "==> [2/3] Verifying PDF regression values against R output"
"$PYTHON" "$HERE/verify_stats.py" "$STRAT_OUT" "$PROC_OUT" --log "$OUT/verification.log"

echo "==> [3/3] Building corrections.pdf + corrections.log"
( cd "$HERE" && "$PYTHON" make_corrections_pdf.py )

echo "==> done"
echo "    (the four replacement figures in new-figs/ are pre-approved and not"
echo "     regenerated here; run 'python make_new_figs.py' to re-derive them.)"
