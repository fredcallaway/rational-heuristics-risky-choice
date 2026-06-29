"""Single source of truth for the *regression* corrections in the marked-up PDF.

The corrected B-values that `make_corrections_pdf.py` writes into the PDF's comment
pane for the two regression sections (Exp1 strategy logistic regressions, p. 913, and
the Exp1 processing-pattern mixed model, p. 915) are defined HERE, once, as structured
records. Both the PDF generator and the verifier import this module, so a number can
never drift between "what the PDF claims" and "what we check against the R output".

Each record carries the `key` of the corresponding row in the standalone R scripts'
output (the bracketed `[strategy ~ param]` / `[feature ~ param]` label), the corrected
`B` value as it must appear, and the literal `comment` string the PDF generator stamps.

These are the ONLY corrections this workflow proves against the R scripts. The chi-square
(p. 919) and participant-kappa (p. 931) corrections are file-reference/copy-paste fixes
that are NOT re-derived here; see corrections/README.md.
"""

# Exp1 strategy logistic regressions -> strategy_regression_standalone.R
# key matches the "[<strategy> ~ <param>]" tag in strategy_regression_results.txt.
STRATEGY = [
    dict(key="SAT_TTB ~ sigma",   B=-0.45, comment="B = -0.45"),
    dict(key="TTB_SAT ~ sigma",   B= 0.26, comment="B = 0.26"),
    dict(key="TTB ~ alpha",       B= 0.61, comment="B = 0.61"),
    dict(key="RandOther ~ alpha", B=-0.60,
         comment="accommodated the increase in TTB by reducing its use of SAT-TTB and "
                 "Targeted Search, participants instead reduced their use of random and "
                 "unclassified strategies (B = -0.60, p < .001; Figure D4)"),
    dict(key="TTB_SAT ~ cost",    B=-0.29, comment="B = -0.29"),
    dict(key="TTB ~ cost",        B=-0.36, comment="B = -0.36"),
    dict(key="SAT_TTB ~ cost",    B= 0.43, comment="B = 0.43"),
]

# Exp1 processing-pattern mixed model -> processing_regression_standalone.R
# key matches the "[<feature> ~ <param>]" tag in processing_regression_results.txt.
PROCESSING = [
    dict(key="processing_pattern ~ alpha", B=-0.084,
         comment=", and they used more attribute-based processing as dispersion "
                 "increased (B = -0.084, p < .001) and"),
]

ALL = STRATEGY + PROCESSING
