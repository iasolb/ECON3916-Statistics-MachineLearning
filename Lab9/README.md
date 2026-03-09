# Recovering Experimental Truths via Propensity Score Matching

## Objective

This project demonstrates how Propensity Score Matching (PSM) can recover a credible causal treatment effect from observational data that, under naive estimation, produces a severely biased and misleading result.

## Methodology

The analysis proceeds through a structured causal inference pipeline applied to the observational subset of the Lalonde (1986) dataset, a canonical benchmark in the program evaluation literature.

- **Diagnosed selection bias** by comparing treated and control groups on pre-treatment covariates, confirming that raw demographic and earnings differences made naive mean-comparison estimates unreliable.
- **Estimated propensity scores** using a logistic regression model trained on observed confounders (age, education, race, marital status, prior earnings, etc.) to predict each individual's probability of receiving the job training treatment.
- **Applied Nearest Neighbor matching** to pair each treated unit with the control unit closest in propensity score, constructing a pseudo-experimental comparison group that approximates the covariate balance of a randomized trial.
- **Evaluated match quality** by reassessing covariate balance post-matching to confirm that systematic differences between groups were substantially reduced.

## Key Findings

| Metric | Estimate |
|---|---|
| Naive observational estimate (ATT) | −$15,204 |
| PSM-adjusted estimate (ATT) | ≈ +$1,800 |
| Experimental benchmark (Lalonde RCT) | ≈ +$1,794 |

The naive comparison of treatment and control means produced an estimate of **−$15,204**, a result that is not only wrong in magnitude but wrong in sign — suggesting the job training program *reduced* earnings. This artifact is driven entirely by selection bias: individuals in the observational control group (drawn from the CPS/PSID) have systematically higher baseline earnings than the treated population.

After propensity score matching, the corrected estimate of the Average Treatment Effect on the Treated (ATT) recovers to approximately **+$1,800**, closely aligning with the experimental benchmark of +$1,794 established by the original randomized evaluation. This represents a correction of nearly $17,000 in estimated impact — a striking demonstration of what rigorous observational methods can achieve when randomization is unavailable.

## Tools

Python · Pandas · Scikit-Learn · Logistic Regression · Nearest Neighbor Matching
