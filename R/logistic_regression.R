rootdir <- "/Users/paulkrueger/Desktop/rational-heuristics-risky-choice/"
outdir <- paste(rootdir,"stats/exp1/dump/1/",sep='')
dat <- read.csv(paste(rootdir,"data/human/1.0/processed/trials.csv",sep=''))

library(lme4)

s1 <- glmer(R_TTB_SAT ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_TTB_SAT.txt",sep=''))
coef(summary(s1))
sink(file = NULL)

s2 <- glmer(R_SAT_TTB ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_SAT_TTB.txt",sep=''))
coef(summary(s2))
sink(file = NULL)

s3 <- glmer(R_TTB ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_TTB.txt",sep=''))
coef(summary(s3))
sink(file = NULL)

s4 <- glmer(R_WADD ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_WADD.txt",sep=''))
coef(summary(s4))
sink(file = NULL)

s5 <- glmer(R_Rand ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_Rand.txt",sep=''))
coef(summary(s5))
sink(file = NULL)

s6 <- glmer(R_Other ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)

sink(file = paste(outdir,"R_Other.txt",sep=''))
coef(summary(s6))
sink(file = NULL)

