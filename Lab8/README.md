## Hypothesis Testing & Causal Evidence Architecture

> *"A model that cannot be falsified is not science — it's storytelling."*

---

### Objective

Most applied data science stops at **estimation** — fitting a model, reading a coefficient, and declaring victory. This project takes a deliberate step back to ask a harder question: *is the signal real, or are we pattern-matching noise?*

Using the **Lalonde (1986) dataset** — a canonical benchmark in program evaluation and causal inference — this lab operationalizes the scientific method as a formal decision framework. The core pivot is from *"What is the effect?"* to *"Can the Null Hypothesis of no effect be credibly rejected?"* This distinction isn't semantic. It is the difference between a finding that survives scrutiny and one that collapses under it.

---

### Technical Approach

- **Parametric Testing via Welch's T-Test (SciPy `ttest_ind`):** Computed the Average Treatment Effect (ATE) of job training on real earnings by framing the problem as a signal-to-noise ratio — the estimated lift divided by the pooled standard error. Welch's formulation was selected over Student's T-Test specifically because it does not assume equal population variances across the treatment and control groups, making it more robust to the heterogeneous subgroups typical in observational labor economics data.

- **Non-Parametric Validation via Permutation Test (10,000 resamples):** Earnings distributions are routinely right-skewed and heavy-tailed — a reality that can quietly invalidate parametric assumptions. To stress-test the T-Test result, a Monte Carlo permutation framework was implemented: treatment labels were randomly shuffled 10,000 times and the ATE was recomputed on each resample, constructing an empirical null distribution from first principles. The observed ATE was then evaluated against this distribution to derive a non-parametric p-value independent of any distributional assumption.

- **Type I Error Control:** Both tests were evaluated against a pre-registered significance threshold (α = 0.05), enforcing a disciplined bound on the false positive rate. Aligning both the parametric and non-parametric results prior to conclusion prevents the implicit p-hacking that occurs when analysts cycle through tests until significance appears.

---

### Key Finding

Both methods converged on the same conclusion: the Null Hypothesis of zero treatment effect is rejected. The job training intervention is associated with a statistically significant lift of approximately **$1,795 in real earnings**, a result validated under parametric, non-parametric, and distributional stress conditions alike. This is *Proof by Statistical Contradiction* — the data is sufficiently inconsistent with a world of no effect that we can confidently discard that world.

---

### Business Insight: Hypothesis Testing as the Safety Valve of the Algorithmic Economy

At production scale, the machinery of modern data science — A/B testing platforms, automated feature selection, multi-armed bandit optimizers — runs fast and largely unsupervised. That velocity is a competitive advantage. It is also a liability.

Without a rigorous falsification layer, the pipeline optimizes for *statistical flukes as readily as it optimizes for genuine signal*. A/B tests run until they're significant. Features are retained because they correlated with the target in one quarter's data. Coefficients are reported without reference to their standard errors. The result is a system that is confidently, systematically wrong — and increasingly difficult to audit as model complexity grows.

Formal hypothesis testing — with pre-specified thresholds, explicit null hypotheses, and non-parametric robustness checks — is the **safety valve** that keeps the pressure honest. It forces the analyst to define what "working" means *before* looking at the data, and to accept "not significant" as a legitimate and informative outcome rather than a failure to be engineered around. In an environment where a single spurious feature in a credit model or a false-positive lift in a recommendation system can propagate to millions of decisions, this discipline is not academic overhead. It is operational risk management.

The Lalonde dataset is a reminder that even well-resourced, well-intentioned policy interventions are subject to this standard. The scientific method is not a bottleneck. It is the architecture that makes trust in algorithmic systems possible.

---

**Tools & Libraries:** Python · SciPy · NumPy · Pandas  
**Dataset:** Lalonde (1986) — National Supported Work Demonstration  
**Methods:** Welch's T-Test · Monte Carlo Permutation Testing · ATE Estimation
