# Architecting the Prediction Engine

**Hedonic OLS Valuation Model · Zillow ZHVI 2026 Micro Dataset**

---

## Objective

Engineer a multivariate ordinary least squares prediction pipeline that forecasts residential real estate valuations from hedonic property attributes, then quantify out-of-sample business risk by converting model loss into an interpretable US-dollar error margin.

## Methodology

- **Data Acquisition & Survey:** Ingested the Zillow ZHVI 2026 Micro Dataset — a cross-sectional snapshot of modern US residential market conditions comprising structural, locational, and institutional quality features (`Square_Footage`, `Property_Age`, `Distance_to_Transit`, `School_District_Rating`).

- **Feature Engineering:** Encoded the categorical `School_District_Rating` variable into a set of binary indicator columns via the Patsy formula interface, establishing a baseline reference category to avoid the dummy-variable trap (perfect multicollinearity).

- **Model Specification & Estimation:** Specified a hedonic pricing equation under the classical linear regression framework and estimated coefficients through ordinary least squares using the `statsmodels` formula API (`smf.ols`). The model regresses `Home_Value` on the full feature set, yielding an R² of **0.766** across 1,000 observations.

- **Predictive Transition:** Shifted the analytical frame from coefficient explanation to predictive generation by extracting the fitted-value vector (`results.predict()`), translating the model from an inferential tool into a forward-looking valuation engine.

- **Loss Quantification:** Calculated the Root Mean Squared Error (RMSE) against observed sale prices, producing a single dollar-denominated performance metric that directly measures the model's expected prediction deviation — the financial error margin an operator would face in a deployment scenario.

- **Residual Diagnostics:** Constructed an interactive residual forensics dashboard (Plotly) plotting fitted values against residual errors, with ±2σ outlier detection, to visually assess heteroscedasticity, non-linearity, and structural breaks in the model's error distribution.

## Key Findings

The hedonic OLS engine explains approximately **76.6%** of the variance in home valuations (R² = 0.766, Adj. R² = 0.765, F = 542.5, p < 0.001). The model's predictive RMSE of **$42,316.69** establishes a concrete financial error bound: on average, the algorithm's valuation deviates from the observed market price by roughly forty-two thousand dollars — a metric that translates statistical performance directly into quantifiable business risk.

Coefficient diagnostics confirm that `Square_Footage` (+$120.79/sq ft) and `Property_Age` (−$814.60/year) are the dominant continuous drivers, while `School_District_Rating` introduces meaningful tier effects relative to the baseline category. The elevated condition number (1.21 × 10⁴) warrants monitoring for multicollinearity in future feature expansions.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3 |
| Data Handling | pandas · NumPy |
| Estimation | statsmodels (Patsy Formula API) |
| Visualization | Plotly Express |

---

*Lab completed March 2026.*