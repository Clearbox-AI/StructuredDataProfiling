import pandas as pd
import copy
import dateparser


class Preprocessor:
    def __init__(self, column_types, n_bins=5):

        self.num = [i for i in column_types.keys() if column_types[i] == 'number'] #x.columns[x.dtypes != "object"]
        self.cat_cols = None
        self.nan_cols = None
        self.date_cols = [i for i in column_types.keys() if column_types[i] == 'string/date']
        self.date_cols_ts = [i for i in column_types.keys() if column_types[i] == 'number/timestamp']
        self.date_cols_ts_ms = [i for i in column_types.keys() if column_types[i] == 'number/timestamp_ms']

        self.bins = n_bins
        self.cat = [i for i in column_types.keys() if column_types[i] in ['number', 'string']]

    def transform(self, x_in):
        x = copy.deepcopy(x_in)
        cat_cols = {}
        xproc = pd.DataFrame()

        num = [i for i in x_in.columns if i in self.num]
        for i in num:
            x[i] = pd.cut(
                x[i], bins=min(self.bins, x[i].nunique()), right=False,
            ).astype(str)

        cat = [i for i in x_in.columns if i in self.cat]
        for i in cat:
            xc = pd.get_dummies(x[i], prefix=i)
            if i+"_nan" in list(xc.columns):
                xc = xc.drop([i+"_nan"], axis=1)
            cat_cols[i] = list(xc.columns)

            xproc = pd.concat([xproc, xc], axis=1)

        # Datetime tests
        date_cols = [i for i in x_in.columns if i in self.date_cols]
        for i in date_cols:

            date_col = [dateparser.parse(str(x[i].iloc[j])) for j in range(x.shape[0])]
            day = [j.weekday() for j in date_col]
            #weekend = [j.weekday() >= 5 for j in date_col]
            month = [j.month for j in date_col]
            hour = [j.hour for j in date_col]

            xc1 = pd.get_dummies(day, prefix=i+'day')
            #xc2 = pd.get_dummies(weekend, prefix=i+'weekend')
            xc3 = pd.get_dummies(month, prefix=i+'month')
            xc4 = pd.get_dummies(hour, prefix=i+'hour')

            if "nan" in list(xc1.columns):
                xc1 = xc1.drop(["nan"], axis=1)

            # if "nan" in list(xc2.columns):
            #     xc2 = xc2.drop(["nan"], axis=1)

            if "nan" in list(xc3.columns):
                xc3 = xc3.drop(["nan"], axis=1)

            if "nan" in list(xc4.columns):
                xc4 = xc4.drop(["nan"], axis=1)

            cat_cols[i] = list(xc1.columns) + list(xc3.columns) + list(xc4.columns)

            xproc = pd.concat([xproc.reset_index(drop=True), xc1, xc3, xc4], axis=1)

        #t1 = dateparser.parse(X['pickup_datetime'].iloc[i])
        self.cat_cols = cat_cols

        return xproc

    def transform_missing(self, x_in):
        x = copy.deepcopy(x_in)
        nan_cols = {}
        xproc = pd.DataFrame()
        cols = self.cat + self.date_cols + self.date_cols_ts + self.date_cols_ts_ms
        num = [i for i in x_in.columns if i in cols]

        for i in num:
            xc = x[i].isna()
            if xc.sum() > 0:
                nan_cols[i] = [i]
                xproc = pd.concat([xproc, xc], axis=1)

        self.nan_cols = nan_cols

        return xproc
