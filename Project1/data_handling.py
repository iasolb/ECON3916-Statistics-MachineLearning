import pandas as pd
import numpy as np


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
        self.full["exper_c"] = (
            self.full["exper"] - self.full["exper"].mean()
        )  # center the experience var
        self.full["expersq"] = (
            self.full["exper_c"] ** 2
        )  # square the centered experience for the quadratic term
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
        self.dependent = self.full[col] if full else self.working[col]
        print(f"Dependent variable set to: {col}")

    def add_independents(self, *cols: str, full: bool = False) -> None:
        df = self.full if full else self.working
        for col in cols:
            self.independents.append(df[col])
        print(f"Independent variables: {[s.name for s in self.independents]}")

    def add_controls(self, *cols: str, full: bool = False) -> None:
        df = self.full if full else self.working
        for col in cols:
            self.controls.append(df[col])
        print(f"Control variables: {[s.name for s in self.controls]}")

    def get_X(self, add_constant: bool = True):
        """Concatenate independents and controls into a single design matrix."""
        import statsmodels.api as sm

        cols = self.independents + self.controls
        X = pd.concat(cols, axis=1)
        return sm.add_constant(X) if add_constant else X

    def get_y(self):
        return self.dependent

    def attach(
        self, col_name: str, series: pd.Series, to_working: bool = False
    ) -> None:
        """Attach a computed column (e.g. IMR) back to full and working subsets."""
        self.full[col_name] = series
        self.working = self.full[self.full["work"].astype(bool)].copy()
        print(f"Attached '{col_name}' to dataset")

    def clear_caches(self) -> None:
        self.dependent = None
        self.independents = []
        self.controls = []
        print("Caches cleared")
