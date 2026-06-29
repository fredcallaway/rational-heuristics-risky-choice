args <- commandArgs(trailingOnly = TRUE)
outdir <- args[1]   # e.g. ../stats/exp1/dump/1/
dat <- read.csv(args[2])   # path to processed human trials.csv

library(lme4)

dat <- dat[dat$cost > 0,]

rank_desc <- function(x) {
    match(x, sort(unique(x), decreasing = TRUE)) - 1
}

dat$sigma <- match(dat$sigma, sort(unique(dat$sigma))) - 1
dat$alpha <- rank_desc(dat$alpha)
dat$cost <- rank_desc(dat$cost)
for (strategy in c("TTB_SAT", "SAT_TTB", "TTB", "WADD", "Rand", "Other")) {
    dat[[strategy]] <- as.integer(dat[[strategy]] == "True")
}

s1 <- glmer(TTB_SAT ~ sigma + alpha + cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_TTB_SAT.txt",sep=''))
coef(summary(s1))
sink(file = NULL)

s2 <- glmer(SAT_TTB ~ sigma + alpha + cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_SAT_TTB.txt",sep=''))
coef(summary(s2))
sink(file = NULL)

s3 <- glmer(TTB ~ sigma + alpha + cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_TTB.txt",sep=''))
coef(summary(s3))
sink(file = NULL)

s4 <- glmer(WADD ~ sigma + alpha + cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_WADD.txt",sep=''))
coef(summary(s4))
sink(file = NULL)

s5 <- glmer(Rand ~ sigma + alpha + cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_Rand.txt",sep=''))
coef(summary(s5))
sink(file = NULL)

s6 <- glmer(Other ~ sigma + alpha + cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_Other.txt",sep=''))
coef(summary(s6))
sink(file = NULL)
