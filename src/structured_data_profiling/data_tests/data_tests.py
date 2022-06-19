from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import copy
import re
from collections import defaultdict
import dateparser
import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools
import datetime
from distfit import distfit


def fit_distributions(x: pd.Series):

    if 10 > x.nunique() > 1:
        dist = distfit(method='discrete')
        dist.fit_transform(x.fillna(0), verbose=False)
        out = ([dist.model['name'], dist.model['score']])
    elif x.nunique() >= 10:
        dist = distfit()
        dist.fit_transform(x.fillna(0), verbose=False)
        out = ([dist.model['name'], dist.model['score']])

    return out


def check_precision(x, tol=1e-8):

    types = ['int16', 'int32', 'float16', 'float32']
    l1 = (x.fillna(0).astype(np.int16) - x.fillna(0)).mean()
    l2 = (x.fillna(0).astype(np.int32) - x.fillna(0)).mean()
    l3 = (x.fillna(0).astype(np.float16) - x.fillna(0)).mean()
    l4 = (x.fillna(0).astype(np.float32) - x.fillna(0)).mean()

    residuals = np.abs([l1, l2, l3, l4]) < tol
    if residuals.sum() > 0:
        val = next((index for index, value in enumerate(residuals) if value != 0), None)
        prec = types[val]
    else:
        prec = 'float64'

    return prec


def check_cardinality(data, threshold=0.005, frac=0.5):
    # Identify columns containing unique value
    unique = {}
    binary = {}
    id_column = []
    high_cardinality = {}
    for i in data.columns:
        n = data[i].nunique()
        if n <= 1:
            unique[i] = data[i].unique()
        if n == 2:
            binary[i] = data[i].unique()
        if n == data[i].shape[0]:
            id_column.append(i)
        elif n > frac*data[i].shape[0]:
            high_cardinality[i] = data[i]

    too_much_info = []
    no_info = []
    categorical_features = data.columns[data.dtypes == 'object']

    cat_features_stats = [
        (
            i,
            data[i].value_counts(),
            data[i].unique(),
            data.columns.get_loc(i),
        )
        for i in categorical_features
    ]

    for column_stats in cat_features_stats:
        if (column_stats[1].shape[0] == 1) or (column_stats[1].shape[0] >= (data.shape[0] * 0.98)):
            no_info.append(column_stats)
        else:
            counts = column_stats[1].values / column_stats[1].values.sum()
            column_name = np.where(counts < threshold)[0]

            if column_name.shape[0] > 0:
                too_much_info.append((column_stats[0], column_stats[1].index[column_name]))

    return unique, binary, id_column, high_cardinality, too_much_info


def find_deterministic_columns_binary(df, binary):
    jtr = np.random.choice(np.arange(df.shape[0]), 5000, replace=True)
    jts = np.random.choice(np.arange(df.shape[0]), 2000, replace=True)
    deterministic = []
    numerical_cols = [i for i in df.columns if df[i].dtype != 'object' if i not in binary]
    for j, i in enumerate(binary):
        clf = DecisionTreeClassifier(max_depth=1).fit(df[numerical_cols].fillna(0).iloc[jtr],
                                                      df[i].fillna(0).iloc[jtr].astype(np.int32))
        score = clf.score(df[numerical_cols].fillna(0).iloc[jts], df[i].fillna(0).iloc[jts].astype(np.int32))
        if score > 0.999:
            deterministic.append([i, clf, copy.deepcopy(numerical_cols)])

    return deterministic, numerical_cols


def find_deterministic_columns_regression(df):
    jtr = np.random.choice(np.arange(df.shape[0]), 1000, replace=True) #TOFIX
    jts = np.random.choice(np.arange(df.shape[0]), 100, replace=True)

    deterministic_num = []
    numerical_cols = [i for i in df.columns if df[i].dtype != 'object']
    if len(numerical_cols) <= 1:
        return deterministic_num, numerical_cols

    for j, i in enumerate(numerical_cols):
        clf = LinearRegression(n_jobs=-1, positive=True, fit_intercept=True).fit(
            df[numerical_cols].fillna(0).drop(i, axis=1).iloc[jtr], df[i].fillna(0).iloc[jtr])

        score = clf.score(df[numerical_cols].fillna(0).drop(i, axis=1).iloc[jts], df[i].fillna(0).iloc[jts])
        if score > 0.9999:
            numerical_cols.remove(i)
            deterministic_num.append([i, clf, copy.deepcopy(numerical_cols)])

    return deterministic_num, numerical_cols


def find_ordinal_columns(df, cat_columns):
    ordinal_dicts = {}

    for i in cat_columns:
        if 2 < df[i].nunique() < 255:
            s = []
            unique = df[i].fillna('N/A').unique()
            for j in unique:
                l = np.array([int(i) for i in re.findall(r'\d+', j.replace('.', ''))])
                if len(l) == 0:
                    l = np.array([-1])

                s.append((l.max() + l.min()) / 2)

            possible_dict = dict(zip([unique[i] for i in np.argsort(s)], np.arange(unique.shape[0])))

            if (np.unique(list(possible_dict.values())).shape[0] / len(list(possible_dict.values())) == 1) and (
                    np.mean(s) > 0):
                ordinal_dicts[i] = possible_dict

    return ordinal_dicts


def identify_dates(data):
    data2 = copy.deepcopy(data)
    possible_dates = []
    a1 = datetime(2000, 1, 1).timestamp()
    a2 = datetime(datetime.now().year + 5, 1, 1).timestamp()
    timestamp = []
    timestamp_ms = []
    # check if format matches timestamp format
    for i in data2.columns:
        if str(data2[i].dtype) not in ['object', 'category']:
            time_column = data2[i].dropna()
            if (time_column > a1).sum() + (time_column < a2).sum() > 1.9 * time_column.shape[0]:
                timestamp.append(i)

    for i in data2.columns:
        if str(data2[i].dtype) not in ['object', 'category']:
            time_column = data2[i].dropna() / 1000
            if (time_column > a1).sum() + (time_column < a2).sum() > 1.9 * time_column.shape[0]:
                data2[i] = data2[i] / 1000
                timestamp_ms.append(i)
    # check if any int is compatible with dates

    for i in data2.columns:
        if str(data2[i].dtype) in ['int', 'int32', 'int64']:
            cond1 = (((data2[i] > -1).astype(int) + (data2[i] < 25).astype(int)) == 2).all()
            cond2 = data2[i].nunique() < 25
            cond3 = (((data2[i] > 0).astype(int) + (data2[i] < 12).astype(int)) == 2).all()
            cond4 = data2[i].nunique() < 13
            cond5 = (((data2[i] > 2000).astype(int) + (data2[i] <= datetime.now().year).astype(int)) == 2).all()
            cond6 = data2[i].nunique() <= (datetime.now().year - 2000)
            cond7 = data2[i].nunique() > 2
            if (cond1 and cond2 and cond7) or (cond3 and cond4 and cond7) or (cond5 and cond6):
                possible_dates.append(i)
        #             print(str(data2[i].iloc[0]), dateparser.parse(str(data2[i].iloc[0])))
        else:
            if dateparser.parse(str(data2[i].iloc[0]), region='EU') is not None:
                possible_dates.append(i)

    possible_dates = possible_dates + timestamp + timestamp_ms
    return possible_dates, timestamp, timestamp_ms


def identify_redundant_dates(data, samples_per_dataframe=40, region='EU'):
    data2 = copy.deepcopy(data)

    idx = np.random.choice(np.arange(data2.shape[0]), samples_per_dataframe, replace=False)
    possible_dates, timestamp, timestamp_ms = identify_dates(data2)
    date_dependency = defaultdict(list)
    for i in possible_dates:
        for j in possible_dates:

            y = data2[j].iloc[idx]
            if not i in [timestamp, timestamp_ms]:

                x = [dateparser.parse(str(data2[i].iloc[w]), region=region) for w in idx]

                x = [w.timestamp() for w in x if w is not None]
            else:
                x = data2[i].iloc[idx].values

            #         date_i = datetime.fromtimestamp(data2[timestamp].iloc[0])
            row_is_dependent = []
            # print(x,y.values)
            if (len(x) == len(y)) and (i != j):
                dates = [datetime.fromtimestamp(w) for w in x]
                for w, y_i in zip(dates, y):
                    date_equivalent = [
                        w.strftime("%Y-%m-%d"),
                        w.strftime("%Y/%m/%d"),
                        w.strftime("%Y-%d-%m"),
                        w.strftime("%Y/%d/%m"),
                        w.strftime("%m-%d-%Y"),
                        w.strftime("%m/%d/%Y"),
                        w.strftime("%d-%m-%Y"),
                        w.strftime("%d/%m/%Y"),
                        w.strftime("%d-%m-%Y 00:00"),
                        w.strftime("%d/%m/%Y 00:00"),
                        w.strftime("%d-%m-%Y %H:%M"),
                        w.strftime("%d/%m/%Y %H:%M"),
                        w.hour,
                        w.day,
                        w.month,
                        int((w.month - 1) / 3) + 1,
                        w.year,
                        w.weekday(),
                        w.weekday(),
                        # w.quarter,
                        # w.week,
                        w.strftime("'%y"),
                        w.strftime("%y"),
                        w.strftime("%Y"),
                        w.strftime("%H:00"),
                        w.strftime("%H:%M"),
                        w.strftime("%H:%M:%S"),
                    ]

                    indexes = [i1 for i1, x in enumerate(date_equivalent) if x == y_i]
                    # print(date_equivalent)
                    # print(i,j,indexes)
                    if indexes:
                        #                            print(i, indexes, date_equivalent[indexes[0]])
                        row_is_dependent += indexes
                for conditions in np.unique(row_is_dependent):
                    row_is_dependent_sub = [x for x in row_is_dependent if x == conditions]
                    print(i, j, row_is_dependent_sub)
                    if len(row_is_dependent_sub) > 0:
                        # print(sum(row_is_dependent_sub)/len(row_is_dependent_sub),row_is_dependent_sub[0])
                        # print((len(row_is_dependent_sub),samples_per_dataframe))
                        # print(i,j)
                        if (sum(row_is_dependent_sub) / len(row_is_dependent_sub) == row_is_dependent_sub[0]) and (
                                len(row_is_dependent_sub) == samples_per_dataframe):
                            print(j, i, row_is_dependent_sub)
                            date_dependency[j, i] = (i, row_is_dependent_sub[0])

    return date_dependency, possible_dates, timestamp + timestamp_ms


def get_features_correlation(X):
    def _cramers_corrected_stat(confusion_matrix):
        """
        Calculate Cramers V statistic for categorial-categorial association,
        using correction from Bergsma and Wicher Journal of the Korean
        Statistical Society 42 (2013): 323-328
        """

        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

    features_correlation = pd.DataFrame(index=X.columns, columns=X.columns)
    for feature_i in X.columns:
        for feature_j in X.columns:
            confusion_matrix = pd.crosstab(X[feature_i], X[feature_j])

            features_correlation[feature_i].loc[feature_j] = \
                float(round(_cramers_corrected_stat(confusion_matrix), 4))

    return features_correlation


def get_label_correlation(Xproc, cat_cols, p_tr=0.75, n_min=100):
    list2d = list(cat_cols.values())
    merged = list(itertools.chain(*list2d))
    corr = []

    for i in merged:
        for j in merged:
            mask = Xproc[j] == 1
            d = Xproc[i][mask].shape[0]
            p = Xproc[i][mask].sum() / d
            sample2 = np.random.choice(np.arange(mask.sum()), mask.sum(), replace=False)
            p2 = Xproc[i].iloc[sample2].sum() / d
            delta = p-p2
            if p > p_tr and i != j and d > n_min and delta > 0.05:
                corr.append((i, j, p, d/Xproc.shape[0]))

    anomalies = []
    for i in corr:
        i1 = np.where((Xproc[i[1]] == 1) & (Xproc[i[0]] == 0))[0]
        if (len(i1) > 0) and (i[2] > 0.99):
            anomalies.append([i[1], i[0], i1])

    return corr, anomalies


