## Main API

```
import pandas as pd
import numpy as np
import statsmodels.api as sm


def calculate_husband_income(husband_wage, husband_hours):
    return husband_wage * husband_hours


class MrozHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.full = (
            pd.read_csv(self.filepath)
            .reset_index()
            .set_index("index", drop=True)
            .rename_axis("ObservationID")
            .drop(columns=["Unnamed: 0"])
        ).rename(
            columns={
                "experience": "exper",
                "educw": "educ",
                "child6": "kidslt6",
                "educwm": "motheduc",
                "educwf": "fatheduc",
            }
        )

        self.full["nwifeinc"] = (
            calculate_husband_income(self.full["wageh"], self.full["hoursh"]) / 1000
        )
        self.full["exper_c"] = self.full["exper"] - self.full["exper"].mean()
        self.full["expersq"] = self.full["exper_c"] ** 2
        self.full["lwage"] = np.log(self.full["hearnw"].replace(0, np.nan))
        self.full["work"] = self.full["work"].map({"yes": 1, "no": 0}).astype(int)
        self.full["city"] = self.full["city"].map({"yes": 1, "no": 0}).astype(int)
        self.working = self.full[self.full["work"].astype(bool)].copy()

        frows, ffeatures = self.full.shape
        wrows, wfeatures = self.working.shape

        print("Full dataset shape")
        print(f"Number of rows: {frows}")
        print(f"Number of features: {ffeatures}")
        print("Working subset shape:")
        print(f"Number of rows: {wrows}")
        print(f"Number of features: {wfeatures}")

        self.dependent = None
        self.independents = []
        self.controls = []

    def set_dependent(self, col: str, full: bool = False) -> None:
        """
        Set the dependent variable from either full or working dataset

        Args: col: column name to set as dependent variable
              full: if True, use the full dataset; otherwise, use the working subset
        Returns: None
        """
        self.dependent = self.full[col] if full else self.working[col]
        print(f"Dependent variable set to: {col}")

    def add_independents(self, *cols: str, full: bool = False) -> None:
        """
        Add independent variables from either full or working dataset
        Args: cols: column names to add as independent variables
              full: if True, use the full dataset; otherwise, use the working subset
        Returns: None
        """
        df = self.full if full else self.working
        for col in cols:
            self.independents.append(df[col])
        print(f"Independent variables: {[s.name for s in self.independents]}")

    def add_controls(self, *cols: str, full: bool = False) -> None:
        """
        Add control variables from either full or working dataset
        Args: cols: column names to add as control variables
              full: if True, use the full dataset; otherwise, use the working subset
        Returns: None
        """
        df = self.full if full else self.working
        for col in cols:
            self.controls.append(df[col])
        print(f"Control variables: {[s.name for s in self.controls]}")

    def get_X(self, add_constant: bool = True):
        """
        Concatenate independents and controls into a single design matrix
        Args: add_constant: if True, add a constant term to the design matrix
        Returns: X design matrix as a DataFrame
        """
        cols = self.independents + self.controls
        X = pd.concat(cols, axis=1)
        return sm.add_constant(X) if add_constant else X

    def get_y(self):
        return self.dependent

    def attach(
        self, col_name: str, series: pd.Series, to_working: bool = False
    ) -> None:
        """
        Attach a computed column (e.g. IMR) back to full or working subsets.
        Args: col_name: name of the column to attach
              series: the Series to attach
              to_working: if True, also update the working subset with the new column
        Returns: None
        """
        self.full[col_name] = series
        if to_working:
            self.working[col_name] = series.loc[self.working.index]
        print(f"Attached '{col_name}' to dataset")

    def get_formula(self) -> str:
        """
        Generate a formula string for use in statsmodels based on current dependent, independents, and controls.
        Returns: formula string
        """
        if self.dependent is None:
            raise ValueError("Dependent variable not set")
        all_vars = self.independents + self.controls
        if not all_vars:
            raise ValueError("No independent or control variables set")
        formula = f"{self.dependent.name} ~ " + " + ".join(s.name for s in all_vars)
        return formula

    def clear_caches(self) -> None:
        """Clear cached dependent, independent, and control variables."""
        self.dependent = None
        self.independents = []
        self.controls = []
        print("Caches cleared")
```

## Full Run 

### Load Data, Calculate IMR

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_handling import MrozHandler

FILEPATH = "data/Mroz.csv"

Mroz = MrozHandler(FILEPATH)

import statsmodels.api as sm

Mroz.set_dependent("work", full=True)
Mroz.add_independents("kidslt6", "nwifeinc", "educ", "agew", "motheduc", "fatheduc", full=True)

probit_result = sm.Probit(Mroz.get_y().astype(int), Mroz.get_X()).fit()
print(probit_result.summary())

```

### Attach Inverse Mills Ratio to dataset (working subset)
```
from scipy.stats import norm
fitted = probit_result.fittedvalues
imr_values = norm.pdf(fitted) / norm.cdf(fitted)
imr_series = pd.Series(imr_values, index=Mroz.full.index)

Mroz.attach("IMR", imr_series, to_working=True)
```

### Stage 2 OLS with attached IMR

```
def heckman_object(Mroz: MrozHandler) -> tuple[pd.Series, pd.DataFrame, MrozHandler]:
    Mroz.clear_caches()
    Mroz.set_dependent("lwage", full=False)
    Mroz.add_independents("exper", "expersq", full=False)
    Mroz.add_controls("educ", "kidslt6", "nwifeinc", "IMR", full=False)
    y = Mroz.get_y().dropna() # Fix the log wage issue by dropping nans, which correspond to non-working individuals
    X = Mroz.get_X().loc[y.index] # Ensure X and y are aligned after dropping nans
    return y, X, Mroz

y, X, _ = heckman_object(Mroz)
ols_result_imr = sm.OLS(y, X).fit()
print(ols_result_imr.summary())
```

### Baseline OLS without IMR

```
def baseline_object(Mroz: MrozHandler) -> tuple[pd.Series, pd.DataFrame, MrozHandler]:
    Mroz.clear_caches()
    Mroz.set_dependent("lwage", full=False)
    Mroz.add_independents("exper", "expersq", full=False)
    Mroz.add_controls("educ", "kidslt6", "nwifeinc", full=False)
    y = Mroz.get_y().dropna()
    X = Mroz.get_X().loc[y.index]
    return y, X, Mroz
    
y, X, BaselineMroz = baseline_object(Mroz)
ols_baseline = sm.OLS(y, X).fit()
print(ols_baseline.summary())
```

### Calculate Peak Experience

```
def calculate_peak_experience(model_result):
    """Calculates the peak of the quadratic experience curve: -b1 / (2 * b2)"""
    b_exper = model_result.params["exper"]
    b_expersq = model_result.params["expersq"]
    
    peak = -b_exper / (2 * b_expersq)
    return peak

# Calculate for both models
peak_log = calculate_peak_experience(ols_result_imr)
peak_lvl = calculate_peak_experience(ols_baseline)

print(f"--- Experience Profile Analysis ---")
print(f"Peak Experience (Log Wage Model): {peak_log:.2f} years")
print(f"Peak Experience (Level Wage Model): {peak_lvl:.2f} years")
```

### Visualizing Diminishing Returns to Experience, and optimal experience

```
def plot_experience_curve(handler, model_result, dep_var_label, ax):
    # 1. Create a range for Experience
    max_exp = handler.working["exper"].max()
    exper_range = np.linspace(0, max_exp, 100)
    
    # 2. Get the average values of all regressors used in the model
    X_means = model_result.model.exog.mean(axis=0)
    X_pred = pd.DataFrame([X_means] * 100, columns=model_result.params.index)
    
    # 3. Update only the experience columns
    X_pred["exper"] = exper_range
    X_pred["expersq"] = exper_range**2
    
    # 4. Predict
    y_pred = model_result.predict(X_pred)
    
    # 5. Plot
    observed = handler.working[dep_var_label].dropna()
    ax.scatter(handler.working["exper"], observed, alpha=0.25, color="steelblue", s=15, label="Observed")
    ax.plot(exper_range, y_pred, color="firebrick", linewidth=2.5, label="Predicted (at means)")
    ax.set_xlabel("Years of Experience")
    ax.legend()

# ==== Execution Block ====

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
print("Plotting experience-earnings curves... \n")
print("Panel 1: Log Wage (Heckman Corrected)\n")

# Panel 1: Log Wage (Heckman Corrected)

y_log, X_log, _ = heckman_object(Mroz)
ols_log = sm.OLS(y_log, X_log).fit()

plot_experience_curve(Mroz, ols_log, "lwage", axes[0])
axes[0].set_ylabel("Log Hourly Earnings", fontsize=11)
axes[0].set_title("Log Earnings Model (Selection Corrected)")

# Panel 2: Levels Wage (Baseline)

print("\nPanel 2: Levels Wage - No IMR (Baseline)\n")
Mroz.set_dependent("hearnw", full=False)
Mroz.add_independents("exper", "expersq", full=False)
Mroz.add_controls("educ", "kidslt6", "nwifeinc", full=False)

y_lvl = Mroz.get_y().dropna()   
X_lvl = Mroz.get_X().loc[y_lvl.index]
ols_levels = sm.OLS(y_lvl, X_lvl).fit()

plot_experience_curve(Mroz, ols_levels, "hearnw", axes[1])
axes[1].set_ylabel("Hourly Earnings ($)", fontsize=11)
axes[1].set_title("Earnings Levels Model", fontsize=12)

axes[0].set_title(f"Log Earnings (Peak: {peak_log:.1f} yrs)", fontsize=12) # Include peak experience in title
axes[1].set_title(f"Earnings Levels (Peak: {peak_lvl:.1f} yrs)", fontsize=12) # Include peak experience in title

fig.suptitle("Experience-Earnings Curves: Marginal Effects at the Mean", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```



# ALL RESULTS IN `Final Report/FinalReport.pdf`

