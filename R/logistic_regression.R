args <- commandArgs(trailingOnly = TRUE)
outdir <- args[1]   # e.g. ../stats/exp1/dump/1/
dat <- read.csv(args[2])   # path to processed human trials.csv

library(lme4)

rank_desc <- function(x) {
    match(x, sort(unique(x), decreasing = TRUE)) - 1
}

rank_asc <- function(x) {
    match(x, sort(unique(x))) - 1
}

prepare_data <- function(df) {
    df$sigma <- rank_asc(df$sigma)
    df$alpha <- rank_desc(df$alpha)
    df$cost <- rank_asc(df$cost)
    for (strategy in c("TTB_SAT", "SAT_TTB", "TTB", "WADD", "Rand", "Other")) {
        df[[strategy]] <- as.integer(df[[strategy]] == "True")
    }
    df
}

fit_model <- function(df, strategy) {
    glmer(as.formula(paste(strategy, "~ sigma + alpha + cost + (1 | pid)")),
          data = df, family = binomial, control = glmerControl(optimizer = "bobyqa"), nAGQ = 10)
}

write_coef <- function(strategy) {
    coefs <- coef(summary(fit_model(dat_full, strategy)))
    cost_coefs <- coef(summary(fit_model(dat_cost, strategy)))
    coefs["cost",] <- cost_coefs["cost",]

    sink(file = paste(outdir, "R_", strategy, ".txt", sep = ""))
    print(coefs)
    sink(file = NULL)
}

dat_full <- prepare_data(dat)
dat_cost <- prepare_data(dat[dat$cost > 0,])
for (strategy in c("TTB_SAT", "SAT_TTB", "TTB", "WADD", "Rand", "Other")) {
    write_coef(strategy)
}
