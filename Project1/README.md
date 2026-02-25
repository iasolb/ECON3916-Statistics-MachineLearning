# Labor Supply Analysis: Selection-Corrected Returns to Experience

## Project Overview

This project investigates whether the returns to labor market experience for married women exhibit diminishing marginal returns. Using the classic Mroz (1987) dataset, we address the inherent selection bias in labor economics: we only observe wages for women who choose to work.

To solve this, we implement a **Heckman Two-Step Selection Model** (Stage 1 Probit → Stage 2 OLS with Inverse Mills Ratio) to obtain unbiased estimates of the experience-wage profile.

---

## Key Research Questions

1. **Selection Bias:** Does the decision to enter the labor force significantly correlate with unobserved wage determinants?
2. **Experience Profile:** Do wages follow a concave path (diminishing returns) or a linear path?
3. **Peak Earnings:** At what point does experience maximize hourly earnings for this cohort?

---

## Technical Implementation

The project is split into two main components:

- `data_handling.py` — An object-oriented data manager (`MrozHandler`) that handles cleaning, feature engineering (e.g., non-wife income calculation), and subsetting.
- `project1.ipynb` — The execution layer containing the Probit selection model, the manual calculation of the Inverse Mills Ratio (IMR), and the final corrected regressions.

### The Heckman 2-Step Logic

**Stage 1 — Selection:** A Probit model estimates the probability of labor force participation based on age, education, children, and non-wife income.

**Inverse Mills Ratio (IMR):** Computed as $\lambda(\hat{z}) = \frac{\phi(\hat{z})}{\Phi(\hat{z})}$, representing the "hazard" of being excluded from the sample.

**Stage 2 — Outcome:** An OLS regression of log-wages on experience and education, including the IMR as a regressor to correct for selection bias.

---

## Key Findings

| Finding | Result |
|---|---|
| IMR Coefficient (λ) | -0.0138 (p=0.957, not significant) |
| Selection Bias Detected | No — baseline OLS estimates are unbiased |
| Experience Effect | Positive and concave (diminishing returns confirmed) |
| exper coefficient | 0.0417 (p=0.002) |
| expersq coefficient | -0.0008 (p=0.049) |

> The insignificant IMR indicates that unobserved factors driving labor force participation are not meaningfully correlated with wages in this sample. The baseline OLS and Heckman-corrected estimates are virtually identical, validating the simpler model.

---

## How to Run

1. Install dependencies: `pip install pandas numpy matplotlib statsmodels scipy`
2. Place `data_handling.py` and `Mroz.csv` in the same directory as the notebook
3. Run `project1.ipynb` to regenerate the full analysis and visualizations

---

## Author

**Ian Solberg**  
ECON3916: Statistics and Machine Learning  
Professor Richeng Piao