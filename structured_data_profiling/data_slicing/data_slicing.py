import copy
import itertools

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def create_interval(string: str):
    bounds = string.split("_")[-1].replace("[", "").replace(")", "").split(",")
    return float(bounds[0]), float(bounds[1])


def feature_importance(data, target, task="classification"):

    X = copy.deepcopy(data)
    X = X.drop(target, axis=1)
    y = data[target]
    y = data[target].astype("category")
    y = y.cat.codes

    num_cols = X.columns[X.dtypes != "object"]
    cat_cols = X.columns[X.dtypes == "object"]
    Xprepro = pd.DataFrame()
    for i in cat_cols:
        Xprepro[i] = X[i].astype("category")
        Xprepro[i] = Xprepro[i].cat.codes

    for i in num_cols:
        Xprepro[i] = X[i]
    if task == "classification":
        clf = RandomForestClassifier(max_depth=1, n_estimators=50).fit(Xprepro, y)
    else:
        clf = RandomForestRegressor(max_depth=1, n_estimators=50).fit(Xprepro, y)

    c = [i for i in zip(Xprepro.columns, clf.feature_importances_)]
    c.sort(key=lambda tup: tup[1])
    return c


def find_slices(X, columns):
    parse_conditions = []

    for i in columns:
        feat2 = i
        if X[i].dtype != "object":
            # intervals = pd.cut(X[feat2],3,right=False).astype(str).unique()
            intervals = list(
                pd.cut(
                    X[feat2],
                    min(X[feat2].nunique(), 3),
                    right=False,
                ).cat.categories.astype(str),
            )
            parse_column = []
            for j in range(len(intervals)):

                interval = create_interval(intervals[j])
                if j == 0:
                    sym = ">="
                else:
                    sym = ">"
                parse_arg = (
                    "`"
                    + feat2
                    + "`"
                    + sym
                    + str(interval[0])
                    + " and "
                    + "`"
                    + feat2
                    + "`<="
                    + str(interval[1])
                )
                parse_column.append(parse_arg)
            parse_conditions.append(parse_column)
        else:
            cats = X[feat2].unique()
            parse_column = []
            for j in cats:
                parse_arg = "`" + feat2 + "`==" + "'" + str(j) + "'"

                parse_column.append(parse_arg)
            parse_conditions.append(parse_column)

    slices = list(itertools.product(*parse_conditions))

    return [" and ".join(slice) for slice in slices]


def check_column_balance(X, target=None):
    IRI = []
    cols = [i for i in X.columns if X[i].nunique() > 1]

    if target:
        cols.remove(target)

    for i in cols:

        slice = find_slices(X, [i])
        s = []
        for w in slice:
            sub = X.query(w)
            if sub.shape[0] > 0:
                s.append(sub.shape[0])
        s = pd.Series(s)
        counts = s / s.sum()
        m = counts.max()
        IR = counts / m
        IRI.append(IR.mean())

    c1 = [i for i in zip(cols, IRI)]
    c1.sort(key=lambda tup: tup[1])
    return c1
