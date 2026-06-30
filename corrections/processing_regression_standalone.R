# Stand-alone reproduction of the Experiment 1 behavioral-feature regressions
# (processing pattern, attribute variance, alternative variance, etc.).
#
# Purpose: independently check the dispersion (alpha^-1) coefficients quoted in
# CHANGES.md, section 2 ("Experiment 1 processing-pattern regression") and the
# unchanged attribute/alternative-variance numbers in the Payne paragraph. This
# reproduces run_statistics.py's `*-<param>_mixedlm.txt` outputs in R so the two
# implementations can be eyeballed side by side.
#
# It mirrors run_statistics.py exactly:
#   - dispersion is alpha^-1: alpha is inverted (1/alpha) BEFORE ranking, then
#     rank-coded ASCENDING, so +1 per step = more dispersion (more peaky).
#   - sigma (stakes) and cost are also 0-indexed ascending condition ranks.
#   - each feature is regressed on ONE predictor at a time (univariate), with a
#     random intercept per participant, as a LINEAR mixed model.
#       smf.mixedlm(p ~ c, df, groups=pid)   <->   lmer(p ~ c + (1 | pid))
#   - rows with NA in the feature are dropped (dropna).
# Unlike strategy_regression_standalone.R, the cost effect here is NOT taken from
# a cost>0 subset; run_statistics.py fits every feature on the full data.
#
# Usage (from anywhere):
#     Rscript --vanilla processing_regression_standalone.R [data.csv] [out.txt]
# Defaults (all relative to this script, staying within corrections/):
#     data.csv -> input/human_trials.csv
#     out.txt  -> output/processing_regression_results.txt

suppressPackageStartupMessages(library(lme4))

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so it runs from any working directory.
# ---------------------------------------------------------------------------
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(this_file)) dirname(normalizePath(this_file)) else getwd()

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else
    file.path(script_dir, "input", "human_trials.csv")
out_path <- if (length(args) >= 2) args[2] else
    file.path(script_dir, "output", "processing_regression_results.txt")

# ---------------------------------------------------------------------------
# Data prep: identical coding to run_statistics.py (lines ~176-184).
# ---------------------------------------------------------------------------
rank_asc <- function(x) match(x, sort(unique(x))) - 1  # 0-indexed ascending rank

prepare_data <- function(df) {
    df$alpha <- 1 / df$alpha          # dispersion is alpha^-1 (do this FIRST)
    df$sigma <- rank_asc(df$sigma)    # higher = higher stakes
    df$alpha <- rank_asc(df$alpha)    # higher = more dispersion (more peaky)
    df$cost  <- rank_asc(df$cost)     # higher = higher cost
    df
}

# Fit the univariate linear mixed model for one feature and one predictor, and
# return the slope row. REML = FALSE to match statsmodels MixedLM (which fits by
# ML by default).
slope_row <- function(df, feature, param) {
    d <- df[!is.na(df[[feature]]), ]
    m <- lmer(as.formula(paste0(feature, " ~ ", param, " + (1 | pid)")),
              data = d, REML = FALSE,
              control = lmerControl(calc.derivs = FALSE))
    cf <- summary(m)$coefficients
    # lme4 reports no p-value by default; derive a Wald z p-value to match the
    # large-sample test statsmodels MixedLM uses.
    est <- cf[param, "Estimate"]
    se  <- cf[param, "Std. Error"]
    z   <- est / se
    p   <- 2 * pnorm(-abs(z))
    list(Estimate = est, SE = se, z = z, p = p, n = nrow(d), n_pid = nlevels(factor(d$pid)))
}

# Format B/p the way run_statistics.py writes the *_mixedlm.txt snippets:
# B to 2 significant figures, "p < 0.001" below 0.001 else "p = <2sigfig>".
fmt <- function(r) {
    p_str <- if (r$p < 0.001) "p < 0.001" else sprintf("p = %.2g", r$p)
    sprintf("B = %.2g, %s", r$Estimate, p_str)
}

# ---------------------------------------------------------------------------
# Fit and report.
# ---------------------------------------------------------------------------
dat <- read.csv(data_path)
dat <- prepare_data(dat)

# Each entry: (feature, predictor) cell, the paper prose it backs, and the value
# currently committed in stats/exp1/2/<feature>-<param>_mixedlm.txt.
reports <- list(
    list(label = "Processing pattern ~ stakes",
         prose = "more alternative-based processing as the stakes increased",
         committed = "B=0.052, p=0.016",
         feature = "processing_pattern", param = "sigma"),
    list(label = "Processing pattern ~ dispersion",
         prose = "more attribute-based processing as dispersion increased (CORRECTED sign)",
         committed = "B=-0.084, p<0.001",
         feature = "processing_pattern", param = "alpha"),
    list(label = "Processing pattern ~ cost",
         prose = "more attribute-based processing as cost increased",
         committed = "B=-0.073, p<0.001",
         feature = "processing_pattern", param = "cost"),
    list(label = "Attribute variance ~ stakes",
         prose = "spread clicks more uniformly across attributes as stakes increase",
         committed = "B=-0.01, p<0.001",
         feature = "click_var_outcome", param = "sigma"),
    list(label = "Attribute variance ~ dispersion",
         prose = "less evenly across attributes as dispersion increases (Payne paragraph)",
         committed = "B=0.0091, p<0.001",
         feature = "click_var_outcome", param = "alpha"),
    list(label = "Attribute variance ~ cost",
         prose = "more discerning across attributes as cost increases",
         committed = "B=0.017, p<0.001",
         feature = "click_var_outcome", param = "cost"),
    list(label = "Alternative variance ~ stakes",
         prose = "spread clicks more uniformly across alternatives as stakes increase",
         committed = "B=-0.004, p=0.0016",
         feature = "click_var_gamble", param = "sigma"),
    list(label = "Alternative variance ~ dispersion",
         prose = "more evenly across alternatives as dispersion increases (Payne paragraph)",
         committed = "B=-0.0026, p<0.001",
         feature = "click_var_gamble", param = "alpha"),
    list(label = "Alternative variance ~ cost",
         prose = "more discerning across alternatives as cost increases",
         committed = "B=0.009, p<0.001",
         feature = "click_var_gamble", param = "cost")
)

con <- file(out_path, open = "wt")
writeLines(c(
    "Experiment 1 behavioral-feature linear mixed regressions",
    "Independent R reproduction of run_statistics.py's *_mixedlm.txt outputs.",
    paste("Data:", normalizePath(data_path)),
    paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    "",
    "Model: lmer(feature ~ predictor + (1 | pid), REML = FALSE), one predictor",
    "at a time. Predictors are 0-indexed condition ranks (one step = +1):",
    "sigma = stakes, alpha = dispersion (1/alpha, ranked so higher = MORE",
    "dispersion / more peaky), cost = cost. p from Wald z test.",
    paste(rep("=", 70), collapse = "")
), con)

for (r in reports) {
    row <- slope_row(dat, r$feature, r$param)
    writeLines(c(
        "",
        sprintf("%s  [%s ~ %s]", r$label, r$feature, r$param),
        sprintf("  prose:     \"%s\"", r$prose),
        sprintf("  committed: %s", r$committed),
        sprintf("  R result:  %s", fmt(row)),
        sprintf("  full:      Estimate=%.5f  SE=%.5f  z=%.3f  p=%.3g  (n=%d, pid=%d)",
                row$Estimate, row$SE, row$z, row$p, row$n, row$n_pid)
    ), con)
}
close(con)

cat("Wrote", out_path, "\n")
