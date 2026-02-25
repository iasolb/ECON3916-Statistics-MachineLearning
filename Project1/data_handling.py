import pandas as pd


class MrozHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.all = (
            pd.read_csv(self.filepath)
            .reset_index()
            .set_index("index", drop=True)
            .rename_axis("ObservationID")
            .drop(columns=["Unnamed: 0"])
        )
        self.working = self.full[self.full['work'] == True]

        rows, features = self.df.shape
        print(f"Number of rows: {rows}")
        print(f"Number of features: {features}")

        self.independent = None
        self.dependent = None
        self.controls = []



    def set_dependent(col: str, full: bool) -> None:
        if full:
            self.dependent = self.full[col]
        else:
            self.dependent = self.working[col]

    def set_independent(col: str, full: bool) -> None:
        if full:
            self.independent = self.full[col]
        else:
            self.independent = self.working[col]

    def add_control(col: str, full: bool) -> None:
        if full:
            self.controls.append(self.full[col])
        else:
            self.controls.append(self.working[col])

    def clear_caches() -> None:
        self.dependent = None
        self.independent = None 
        self.controls = []



    

