#!/usr/bin/env python
"""Prove the regression corrections in the PDF are consistent with the R script output.

The corrected B-values that make_corrections_pdf.py stamps into corrections.pdf come from
regression_corrections.py. This script re-derives the *same* coefficients independently
by parsing the output of the two standalone R scripts, and asserts that each corrected
value rounds to the R estimate. If any value disagrees, it exits non-zero so the whole
workflow fails loudly.

Usage:
    python verify_stats.py <strategy_results.txt> <processing_results.txt> [--log FILE]

Each R output file has, per coefficient, a block:
    <label>  [<strategy/feature> ~ <param>]
      ...
      full:  Estimate=-0.4517  SE=...  ...
The "[... ~ ...]" tag is the join key with regression_corrections.py.
"""

import argparse
import re
import sys

import regression_corrections as rc

# Matches "[SAT_TTB ~ sigma]" and the following "Estimate=-0.4517" anywhere below it.
KEY_RE = re.compile(r"\[([A-Za-z_]+ ~ [A-Za-z_]+)\]")
EST_RE = re.compile(r"Estimate=(-?\d+\.\d+)")


def parse_estimates(text):
    """Map each '[key]' tag in an R output file to its parsed Estimate (float)."""
    out = {}
    current = None
    for line in text.splitlines():
        m = KEY_RE.search(line)
        if m:
            current = m.group(1)
            continue
        m = EST_RE.search(line)
        if m and current is not None:
            out[current] = float(m.group(1))
            current = None
    return out


def n_decimals(x):
    """Number of decimal places in the *displayed* correction value (e.g. -0.084 -> 3)."""
    s = repr(abs(x))
    return len(s.split(".")[1]) if "." in s else 0


def verify(estimates, log_lines):
    """Check every regression record against the R estimates. Returns list of failures."""
    failures = []
    for rec in rc.ALL:
        key, B = rec["key"], rec["B"]
        if key not in estimates:
            failures.append(f"{key}: no Estimate found in R output")
            log_lines.append(f"  FAIL  {key:<26}  (missing from R output)")
            continue
        est = estimates[key]
        rounded = round(est, n_decimals(B))
        ok = rounded == B
        status = "ok  " if ok else "FAIL"
        log_lines.append(
            f"  {status}  {key:<26}  PDF B={B:+.3g}  R Estimate={est:+.5f}"
            f"  (rounds to {rounded:+.3g})"
        )
        if not ok:
            failures.append(f"{key}: PDF B={B} != round(R Estimate {est}) = {rounded}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy_results")
    ap.add_argument("processing_results")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    estimates = {}
    for path in (args.strategy_results, args.processing_results):
        with open(path) as f:
            estimates.update(parse_estimates(f.read()))

    log_lines = [
        "regression-correction verification",
        f"strategy R output:   {args.strategy_results}",
        f"processing R output: {args.processing_results}",
        f"records checked: {len(rc.ALL)}",
        "-" * 78,
    ]
    failures = verify(estimates, log_lines)
    log_lines.append("-" * 78)
    log_lines.append("RESULT: PASS" if not failures else f"RESULT: FAIL ({len(failures)})")

    report = "\n".join(log_lines)
    print(report)
    if args.log:
        with open(args.log, "w") as f:
            f.write(report + "\n")

    if failures:
        print("\nVERIFICATION FAILED:", file=sys.stderr)
        for fl in failures:
            print("  " + fl, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
