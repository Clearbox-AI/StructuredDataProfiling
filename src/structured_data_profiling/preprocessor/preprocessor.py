import pandas as pd
import copy


def drop_nan(x_in):
    cols = []
    for i in x_in.columns:
        if i.split('_')[-1] != 'nan':
            cols.append(i)

    return x_in[cols]


class Preprocessor:

    def __init__(self, x: pd.DataFrame, n_bins=5):
        self.num = x.columns[x.dtypes != 'object']
        self.cat_cols = None
        self.bins = n_bins
        self.cat = x.columns

    def transform(self, x_in):
        x = copy.deepcopy(x_in)
        cat_cols = {}
        xproc = pd.DataFrame()

        for i in self.num:
            if x[i].nunique() > 10:
                x[i] = pd.cut(x[i], bins=min(self.bins, x[i].nunique()), right=False).astype(str)

        for i in self.cat:
            xc = pd.get_dummies(x[i], prefix=i)
            cat_cols[i] = list(xc.columns)

            xproc = pd.concat([xproc, xc], axis=1)

        self.cat_cols = cat_cols

        return xproc

