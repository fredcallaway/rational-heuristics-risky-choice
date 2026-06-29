# Stand-alone reproduction of the Experiment 1 strategy logistic regressions.
#
# This script reproduces the regression coefficients quoted in CHANGES-alt.md,
# section 2 ("Experiment 1 strategy-regression statistics"). It is functionally
# the same model as code/R/logistic_regression.R, but instead of writing one
# dump file per strategy it writes a SINGLE annotated text file in which each
# reported number is labeled with the exact prose it backs in CHANGES-alt.md,
# so the two can be eyeballed side by side.
#
# Usage (from anywhere):
#     Rscript --vanilla strategy_regression_standalone.R [data.csv] [out.txt]
# Defaults:
#     data.csv -> ../data/human/1.0/processed/trials.csv  (relative to this script)
#     out.txt  -> strategy_regression_results.txt         (next to this script)

suppressPackageStartupMessages(library(lme4))

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so it runs from any working directory.
# ---------------------------------------------------------------------------
this_file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script_dir <- if (length(this_file)) dirname(normalizePath(this_file)) else getwd()

args <- commandArgs(trailingOnly = TRUE)
data_path <- if (length(args) >= 1) args[1] else
    file.path(script_dir, "..", "data", "human", "1.0", "processed", "trials.csv")
out_path <- if (length(args) >= 2) args[2] else
    file.path(script_dir, "strategy_regression_results.txt")

# ---------------------------------------------------------------------------
# Data prep: identical coding to logistic_regression.R.
#   sigma (stakes):     rank ascending  -> +1 per step up in stakes
#   alpha (dispersion): rank DESCENDING -> +1 per step up in dispersion (1/alpha)
#   cost:               rank ascending  -> +1 per step up in cost
# The descending rank on alpha is what makes the alpha coefficient an effect of
# increasing dispersion, matching the corrected interpretation in CHANGES-alt.
# ---------------------------------------------------------------------------
rank_desc <- function(x) match(x, sort(unique(x), decreasing = TRUE)) - 1
rank_asc  <- function(x) match(x, sort(unique(x))) - 1

prepare_data <- function(df) {
    df$sigma <- rank_asc(df$sigma)
    df$alpha <- rank_desc(df$alpha)
    df$cost  <- rank_asc(df$cost)
    for (strategy in c("TTB_SAT", "SAT_TTB", "TTB", "WADD", "Rand", "Other")) {
        df[[strategy]] <- as.integer(df[[strategy]] == "True")
    }
    df
}

fit_model <- function(df, strategy) {
    glmer(as.formula(paste(strategy, "~ sigma + alpha + cost + (1 | pid)")),
          data = df, family = binomial,
          control = glmerControl(optimizer = "bobyqa"), nAGQ = 10)
}

# Return the coefficient row for one predictor. The cost effect is taken from the
# cost>0 subset (matching logistic_regression.R, which substitutes the cost row
# from a model fit on cost-bearing trials only).
coef_row <- function(strategy, param) {
    if (param == "cost") {
        cf <- coef(summary(fit_model(dat_cost, strategy)))
    } else {
        cf <- coef(summary(fit_model(dat_full, strategy)))
    }
    cf[param, ]
}

# Format a coefficient the way the paper/CHANGES-alt reports it: B to 2 sig figs,
# p as "< .001" when below 0.001, otherwise "= .###".
fmt <- function(row) {
    B <- row[["Estimate"]]
    p <- row[["Pr(>|z|)"]]
    p_str <- if (p < 0.001) "p < .001" else sprintf("p = %.3g", p)
    sprintf("B = %.2g, %s", B, p_str)
}

# ---------------------------------------------------------------------------
# Fit and report.
# ---------------------------------------------------------------------------
dat <- read.csv(data_path)
dat_full <- prepare_data(dat)
dat_cost <- prepare_data(dat[dat$cost > 0, ])

# Each entry: the (strategy, predictor) cell and the CHANGES-alt prose it backs.
reports <- list(
    list(label = "Stakes effect on SAT-TTB",
         prose = "stakes had a significant negative effect on the frequency of SAT-TTB",
         strategy = "SAT_TTB", param = "sigma"),
    list(label = "Stakes effect on Targeted Search",
         prose = "a significant positive effect on the frequency of targeted search",
         strategy = "TTB_SAT", param = "sigma"),
    list(label = "Dispersion effect on TTB",
         prose = "Our participants confirmed this prediction (... middle column of Figure 4)",
         strategy = "TTB", param = "alpha"),
    list(label = "Dispersion effect on SAT-TTB",
         prose = "participants showed an increase in both strategies (SAT-TTB: ...)",
         strategy = "SAT_TTB", param = "alpha"),
    list(label = "Dispersion effect on Targeted Search",
         prose = "participants showed an increase in both strategies (... Targeted Search: ...)",
         strategy = "TTB_SAT", param = "alpha"),
    list(label = "Cost effect on Targeted Search",
         prose = "decreasing the use of both targeted search",
         strategy = "TTB_SAT", param = "cost"),
    list(label = "Cost effect on TTB",
         prose = "and TTB",
         strategy = "TTB", param = "cost"),
    list(label = "Cost effect on SAT-TTB",
         prose = "increasing the use of the most frugal strategy, SAT-TTB",
         strategy = "SAT_TTB", param = "cost")
)

con <- file(out_path, open = "wt")
writeLines(c(
    "Experiment 1 strategy logistic regressions",
    "Reproduction of the statistics in CHANGES-alt.md, section 2.",
    paste("Data:", normalizePath(data_path)),
    paste("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
    "",
    "Model: glmer(strategy ~ sigma + alpha + cost + (1 | pid), family = binomial).",
    "Predictors are 0-indexed condition ranks (one step = +1): sigma = stakes,",
    "alpha = dispersion (1/alpha, ranked so higher = more dispersion), cost = cost.",
    "Cost effects are from the cost>0 subset, matching logistic_regression.R.",
    paste(rep("=", 70), collapse = "")
), con)

for (r in reports) {
    row <- coef_row(r$strategy, r$param)
    writeLines(c(
        "",
        sprintf("%s  [%s ~ %s]", r$label, r$strategy, r$param),
        sprintf("  prose:    \"%s\"", r$prose),
        sprintf("  reported: %s", fmt(row)),
        sprintf("  full:     Estimate=%.4f  SE=%.4f  z=%.3f  p=%.3g",
                row[["Estimate"]], row[["Std. Error"]],
                row[["z value"]], row[["Pr(>|z|)"]])
    ), con)
}
close(con)

cat("Wrote", out_path, "\n")
