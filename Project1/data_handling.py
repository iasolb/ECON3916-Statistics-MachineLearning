import pandas as pd


class MrozHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = (
            pd.read_csv(self.filepath)
            .reset_index()
            .set_index("index", drop=True)
            .rename_axis("ObservationID")
            .drop(columns=["Unnamed: 0"])
        )
        rows, features = self.df.shape
        print(f"Number of rows: {rows}")
        print(f"Number of features: {features}")

    def get_subset():
        pass
