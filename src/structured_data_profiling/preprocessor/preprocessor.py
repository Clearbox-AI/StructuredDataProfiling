import pandas as pd
import copy


class Preprocessor:
    def __init__(self, x: pd.DataFrame, n_bins=5):
        self.num = x.columns[x.dtypes != "object"]
        self.cat_cols = None
        self.nan_cols = None

        self.bins = n_bins
        self.cat = x.columns

    def transform(self, x_in):
        x = copy.deepcopy(x_in)
        cat_cols = {}
        xproc = pd.DataFrame()

        for i in self.num:
            x[i] = pd.cut(
                x[i], bins=min(self.bins, x[i].nunique()), right=False,
            ).astype(str)

        for i in self.cat:
            xc = pd.get_dummies(x[i], prefix=i)
            if "nan" in list(xc.columns):
                xc = xc.drop(["nan"], axis=1)
            cat_cols[i] = list(xc.columns)

            xproc = pd.concat([xproc, xc], axis=1)

        self.cat_cols = cat_cols

        return xproc

    def transform_missing(self, x_in):
        x = copy.deepcopy(x_in)
        nan_cols = {}
        xproc = pd.DataFrame()

        for i in self.cat:
            xc = x[i].isna()
            if xc.sum() > 0:
                nan_cols[i] = [i]
                xproc = pd.concat([xproc, xc], axis=1)

        self.nan_cols = nan_cols

        return xproc
