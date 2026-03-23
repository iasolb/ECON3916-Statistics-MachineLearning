# The Architecture of Dimensionality: Hedonic Pricing & the FWL Theorem

## Objective

This project implements a multivariate hedonic pricing model on 2026 California residential real estate data to empirically demonstrate how the Frisch-Waugh-Lovell (FWL) theorem mechanically eliminates omitted variable bias by isolating the partial effect of each regressor through residual decomposition.

## Methodology

**Data**: Zillow synthetic dataset comprising California residential transactions with three core features — `Sale_Price`, `Property_Age`, and `Distance_to_Tech_Hub`.

**Technical Stack**: Python 3.10+, pandas, statsmodels.formula.api, matplotlib.

**Procedure**:

- Estimated a bivariate OLS regression of `Sale_Price` on `Property_Age` alone, establishing the naïve (omitted-variable) baseline coefficient.
- Extended the specification to a full multivariate model regressing `Sale_Price` on both `Property_Age` and `Distance_to_Tech_Hub`, producing the ceteris paribus partial effects.
- Manually executed the FWL partialling-out procedure: regressed `Property_Age` on `Distance_to_Tech_Hub` and extracted the residuals, then regressed `Sale_Price` on `Distance_to_Tech_Hub` and extracted those residuals, and finally regressed the sale-price residuals on the property-age residuals.
- Confirmed numerical equivalence between the FWL residual-on-residual coefficient and the multivariate OLS coefficient on `Property_Age`, validating the theorem to machine precision.

## Key Findings

The bivariate specification exhibited severe omitted variable bias. Excluding `Distance_to_Tech_Hub` — a variable negatively correlated with both property age and sale price in California's tech-corridor markets — caused the OLS estimator to falsely attribute inflated explanatory power to `Property_Age`. The naïve coefficient overstated the age premium by absorbing geographic demand effects that properly belong to tech-hub proximity.

Introducing `Distance_to_Tech_Hub` into the regression corrected this distortion. The FWL residual-on-residual estimate produced an exact match to the multivariate coefficient, confirming that OLS achieves ceteris paribus by algebraically stripping shared covariance from each regressor before estimating its partial slope. The exercise demonstrates that multivariate regression is not merely a statistical convenience — it is a mechanical application of orthogonal projection that partitions the variance structure of the design matrix to isolate conditionally independent variation.
