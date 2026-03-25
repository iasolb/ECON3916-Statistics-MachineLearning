# Project 1: The Econometric Foundation

**ECON 3916: Statistical Machine Learning for Economics**
Ian Solberg | Professor Richeng Piao | Spring 2026

---

## Research Question

Do returns to labor market experience diminish at higher wage levels for married women, after correcting for sample selection bias?

This project estimates a Heckman (1979) selection-corrected Mincer wage equation using the Mroz (1987) dataset (N = 753 married women, 428 with observed wages) from the 1975 Panel Study of Income Dynamics.

## Repository Structure

```
Project1/
├── Final Report/           # Submitted .docx report and code appendix
├── Proposal/               # Original research proposal (PDF)
├── assets/                 # Figures and visualizations
├── data/                   # Mroz.csv dataset
├── data_handling.py        # MrozHandler class (data cleaning, variable construction, model API)
├── data_summary.ipynb      # Summary statistics and Table 1 generation
├── phase2.ipynb            # Phase 2: Data audit, EDA, distribution analysis
├── phase3.ipynb            # Phase 3: Baseline Heckman model, interaction tests, ID strategy
├── phase4_checks.ipynb     # Phase 4: Robustness checks (naive OLS, outliers, city, exclusion restrictions)
├── phase4_fullrun.ipynb    # Phase 4: Full model execution with HC1 robust errors
└── README.md
```

## Methodology

The identification strategy follows Heckman's two-step procedure:

1. **Stage 1 (Selection):** A probit model estimates labor force participation using `kidslt6`, `nwifeinc`, `educ`, `agew`, `motheduc`, and `fatheduc`. Mother's and father's education serve as exclusion restrictions. The Inverse Mills Ratio (IMR) is computed from the fitted values.

2. **Stage 2 (Outcome):** OLS regresses `lwage` on `exper`, `expersq`, `educ`, `kidslt6`, `nwifeinc`, and the IMR using the 428 working women. All standard errors are HC1 heteroscedasticity-robust.

## Key Findings

- **Experience:** An additional year is associated with a 2.5% wage increase (p = 0.001), with a negative quadratic term confirming diminishing returns.
- **Education:** Returns of approximately 9.8% per year (p = 0.001), though likely upward biased due to omitted ability.
- **Selection bias:** The IMR is insignificant across all specifications (p = 0.962 in the baseline), suggesting minimal selection bias in this sample.
- **Robustness:** Results are stable across naive OLS, IQR-trimmed samples, urban residence controls, and alternative exclusion restriction sets.

## Project Phases

| Phase | Deliverable | Description |
|-------|------------|-------------|
| 1 | `Proposal/` | Research question, variable map, hypothesis, empirical strategy |
| 2 | `phase2.ipynb` | Data audit, cleaning pipeline, distributional analysis, EDA |
| 3 | `phase3.ipynb` | Baseline model, interaction terms, identification police meeting |
| 4 | `Final Report/` | 10-page narrative, Code Appendix|

## Data

**Source:** Mroz, T.A. (1987). Distributed via Wooldridge (2002) and the R `wooldridge` package.

**Key variables:**

| Variable | Description |
|----------|-------------|
| `lwage` | Log hourly wage (observed for workers only) |
| `exper`, `expersq` | Years of experience and centered quadratic |
| `educ` | Years of education |
| `kidslt6` | Number of children under 6 |
| `nwifeinc` | Non-wife household income ($000s) |
| `motheduc`, `fatheduc` | Parental education (exclusion restrictions) |
| `IMR` | Inverse Mills Ratio (computed from Stage 1) |

## `data_handling.py`

The `MrozHandler` class centralizes all data cleaning and model setup:

- Loads and renames columns to standard Wooldridge conventions
- Constructs derived variables (`nwifeinc`, `exper_c`, `expersq`, `lwage`)
- Recodes binary indicators (`work`, `city`)
- Maintains `full` (N = 753) and `working` (N = 428) subsets
- Provides a caching API for dependent, independent, and control variable assignment across model specifications

## References

- Heckman, J.T. (1979). Sample Selection Bias as a Specification Error. *Econometrica*, 47(1), 153-161.
- Mincer, J. (1974). *Schooling, Experience, and Earnings*. NBER.
- Mroz, T.A. (1987). The Sensitivity of an Empirical Model of Married Women's Hours of Work to Economic and Statistical Assumptions. *Econometrica*, 55(4), 765-799.
- Wooldridge, J.M. (2002). *Introductory Econometrics: A Modern Approach*. South-Western.