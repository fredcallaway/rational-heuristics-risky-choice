dat <- read.csv("../data/human/1.0/processed/trials.csv")

# dat[TTB_SAT] <- (dat$TTB_SAT-mean(dat$TTB_SAT))/sd(dat$TTB_SAT)
# dat[SAT_TTB] <- (dat$SAT_TTB-mean(dat$SAT_TTB))/sd(dat$SAT_TTB)
# dat[TTB] <- (dat$TTB-mean(dat$TTB))/sd(dat$TTB)
# dat[WADD] <- (dat$WADD-mean(dat$WADD))/sd(dat$WADD)
# dat[Rand] <- (dat$Rand-mean(dat$Rand))/sd(dat$Rand)
# dat[Other] <- (dat$Other-mean(dat$Other))/sd(dat$Other)

# hdp <- within(dat, {
#   Married <- factor(Married, levels = 0:1, labels = c("no", "yes"))
#   DID <- factor(DID)
#   HID <- factor(HID)
#   CancerStage <- factor(CancerStage)
# })
library(lme4)

# mod <- glmer(strategy ~ sigma + alpha + cost + (1 | pid),
#              data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#            nAGQ = 10)

# print(mod)


# mod2 <- glmer(strategy ~ sigma + (1 | pid),
#              data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#              nAGQ = 10)

# s1 <- glmer(TTB_SAT ~ R_sigma + R_alpha + R_cost + (1 | pid),
#              data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#              nAGQ = 10)
s2 <- glmer(R_SAT_TTB ~ R_sigma + R_alpha + R_cost + (1 | pid),
            data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
            nAGQ = 10)
sink(file = '../stats/exp1/R_SAT_TTB.txt')
summary(s2)
sink(file = NULL)
# s3 <- glmer(TTB ~ R_sigma + R_alpha + R_cost + (1 | pid),
#             data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#             nAGQ = 10)
# s4 <- glmer(WADD ~ R_sigma + R_alpha + R_cost + (1 | pid),
#             data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#             nAGQ = 10)
# s5 <- glmer(Rand ~ R_sigma + R_alpha + R_cost + (1 | pid),
#             data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#             nAGQ = 10)
# s6 <- glmer(Other ~ R_sigma + R_alpha + R_cost + (1 | pid),
#             data = dat, family = binomial, control = glmerControl(optimizer = "bobyqa"),
#             nAGQ = 10)


# hdp <- read.csv("https://stats.idre.ucla.edu/stat/data/hdp.csv")






