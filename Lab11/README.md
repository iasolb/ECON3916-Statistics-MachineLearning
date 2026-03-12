# Data Wrangling & Engineering Pipeline

## Objective

To engineer a robust feature preparation pipeline that transforms a chaotic human resources economics dataset into a clean, model-ready structure suitable for downstream econometric inference.

## Methodology

- **Missingness Diagnostics:** Profiled the dataset using `missingno` to visualize and classify missing data patterns, confirming a Missing At Random (MAR) structure that informed the choice of imputation strategy.
- **Missing Value Imputation:** Applied conditional imputation techniques via `pandas` to fill gaps in a manner consistent with the underlying data-generating process, preserving distributional properties for unbiased estimation.
- **Dummy Variable Encoding with Reference Class Dropping:** Converted categorical variables into binary indicators using one-hot encoding, then deliberately dropped one reference category per variable to avoid perfect multicollinearity (the Dummy Variable Trap), ensuring full rank in the design matrix for OLS identification.
- **Target Encoding for High-Cardinality Features:** Compressed geographic variables with excessive unique values using `category_encoders.TargetEncoder`, replacing sparse categorical levels with their conditional mean of the target variable — a dimensionality reduction technique that retains predictive signal without inflating the feature space.

## Key Findings

The pipeline successfully resolved three core structural challenges in applied econometric data preparation. First, missingness was diagnosed as MAR rather than MCAR or MNAR, which validated the use of conditional imputation over listwise deletion and preserved sample size for efficiency gains. Second, explicit reference class dropping eliminated the rank deficiency that would otherwise cause `statsmodels` OLS to fail or produce uninterpretable coefficients. Third, target encoding reduced geographic feature dimensionality by orders of magnitude while maintaining the informational content relevant to the dependent variable, producing a parsimonious specification ready for regression analysis.
