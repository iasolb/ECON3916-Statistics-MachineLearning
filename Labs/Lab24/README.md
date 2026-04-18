# Causal ML — Double Machine Learning for 401(k) Policy Evaluation

## Objective

This project applies Double Machine Learning (DML) to obtain a debiased, root-N consistent estimate of the causal effect of 401(k) eligibility on household net financial assets, while leveraging flexible nonparametric nuisance models that would otherwise introduce regularization bias in a naïve plug-in approach.

## Methodology

- **Regularization Bias Demonstration.** Simulated a partially linear data-generating process (true ATE = 5.0) and showed that naïve LASSO regression shrinks the treatment coefficient toward zero, motivating the need for orthogonal/debiased estimation.
- **Double Machine Learning (PLR).** Implemented the Partially Linear Regression framework of Chernozhukov et al. (2018) via the `DoubleML` package. Random Forest learners were used for both the outcome and treatment nuisance models, with 5-fold cross-fitting to avoid overfitting bias.
- **Average Treatment Effect Estimation.** Estimated the ATE of 401(k) plan eligibility on net financial assets using the full observational sample, with heteroskedasticity-robust standard errors and confidence intervals.
- **Conditional ATE Analysis.** Stratified the sample by income quartile and re-estimated the PLR model within each subgroup to characterize treatment effect heterogeneity across the income distribution.
- **Sensitivity Analysis.** Applied the DML sensitivity analysis framework (Chernozhukov et al., 2022) to assess robustness of the ATE to potential unmeasured confounders, with contour plots over partial R² bounds.

## Key Findings

- **Regularization Bias.** LASSO underestimated the treatment effect in the simulated setting (true ATE = 5.0), confirming that standard penalized regression is inappropriate for causal coefficient recovery without debiasing.
- **ATE Estimate.** 401(k) eligibility increases net financial assets by approximately $____ (95% CI: $____ – $____), statistically significant at the 1% level.
  <!-- ▸ Replace the blanks above with your point estimate and confidence interval. -->
- **Heterogeneity by Income.**
  <!-- ▸ Replace the placeholders below with your CATE estimates per quartile. -->
  | Income Quartile | CATE Estimate | 95% CI |
  |:---------------:|:-------------:|:------:|
  | Q1 (lowest)     | $____         | ____   |
  | Q2              | $____         | ____   |
  | Q3              | $____         | ____   |
  | Q4 (highest)    | $____         | ____   |

  Treatment effects [increased / were relatively stable / showed an inverted-U pattern] across the income distribution, suggesting that ____.
- **Sensitivity.** The robustness value exceeded ____, indicating that an omitted confounder would need to explain an implausibly large share of residual variation in both treatment and outcome to nullify the estimated effect.

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Poterba, J., Venti, S., & Wise, D. (1995). Do 401(k) contributions crowd out other personal saving? *Journal of Public Economics*, 58(1), 1–32.