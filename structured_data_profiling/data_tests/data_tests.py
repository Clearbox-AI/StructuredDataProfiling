import copy
import itertools
import re

from distfit import distfit
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier  # ,export_text
from tqdm import tqdm


def is_text(x: pd.Series):
    """Checks whether column contains nominal labels or free text

    Args:
        x (pd.Series): Dataframe column containing strings

    Returns:
        Bool: True if column contains free text
    """
    len_el = [
        len(j) for j in x.fillna("nan").sample(min(x.shape[0], 1000), replace=False)
    ]
    return (
        (x.value_counts().values.mean() < 1.5)
        & (np.max(len_el) / np.min(len_el) > 10.0)
        & (np.max(len_el) > 30.0)
    )


def fit_distributions(x: pd.Series):
    """This function uses the distfit to fit a probability distribution
        for a given column
    Args:
        x (pd.Series): Column containing samples used to fit the distribution

    Returns:
        List: A list containing the best distribution and the corresponding confindence.
    """
    if 10 > x.nunique() > 1:
        dist = distfit(method="discrete")
        dist.fit_transform(x.fillna(0), verbose=False)
        out = [dist.model["name"], dist.model["score"]]
    elif x.nunique() >= 10:
        dist = distfit()
        dist.fit_transform(x.fillna(0), verbose=False)
        out = [dist.model["name"], dist.model["score"]]

    return out


def check_precision(x: pd.Series, tol=1e-8):
    """This function determines the numerical representation needed to represent a
    certain column.

    Args:
        x pd.Series: a Pandas Series
        tol (float, optional): A tolerance representing an acceptable information loss.
        Defaults to 1e-8.

    Returns:
        string: the numerical representation chosen between ["int16", "int32", "float16"
        , "float32"]
    """
    types = ["int16", "int32", "float16", "float32"]
    l1 = (x.fillna(0).astype(np.int16) - x.fillna(0)).mean()
    l2 = (x.fillna(0).astype(np.int32) - x.fillna(0)).mean()
    l3 = (x.fillna(0).astype(np.float16) - x.fillna(0)).mean()
    l4 = (x.fillna(0).astype(np.float32) - x.fillna(0)).mean()

    residuals = np.abs([l1, l2, l3, l4]) < tol
    if residuals.sum() > 0:
        val = next((index for index, value in enumerate(residuals) if value != 0), None)
        prec = types[val]
    else:
        prec = "float64"

    return prec


def find_deterministic_columns_binary(df, binary):
    """This test determines whether any of the columns of a given dataframe
    can be defined deterministically using a condition such as:
    A = C > t
    Where C is another numerical column from the same dataframe and t a generic
    threshold.

    Args:
        df (pd.DataFrame): a pandas Dataframe containing a number of binary columns
        binary (List): a list containing the names of the binary columns present in the
        dataframe

    Returns:
        _type_: _description_
    """
    jtr = np.random.choice(np.arange(df.shape[0]), 5000, replace=True)
    jts = np.random.choice(np.arange(df.shape[0]), 2000, replace=True)
    deterministic = []

    numerical_cols = [
        i for i in df.columns if df[i].dtype != "object" if i not in binary
    ]
    for j, i in enumerate(binary):
        y = df[i].astype("category").cat.codes
        clf = DecisionTreeClassifier(max_depth=1).fit(
            df[numerical_cols].fillna(0).iloc[jtr],
            y.iloc[jtr].astype(np.int32),
        )
        score = clf.score(
            df[numerical_cols].fillna(0).iloc[jts],
            y.iloc[jts].astype(np.int32),
        )

        # text = export_text(clf, feature_names=list(df[numerical_cols].columns))
        # print(export_text(clf, decimals=4))
        # print(i, text, score)
        #
        # print(float(text.split('|')[-3].split('>')[-1]))
        # print(float(text.split('|')[-1].split(':')[-1]))

        if score > 0.999:
            deterministic.append([i, clf, copy.deepcopy(numerical_cols)])

    return deterministic, numerical_cols


def find_linear_combinations(df: pd.DataFrame):
    """This function finds whether any of the columns of the input dataframe
    can be expressed as a linear combination of other columns. This is done by fitting
    a number of linear regressors.

    Args:
        df (pd.DataFrame): an input pandas dataframe

    Returns:
        List: A list of tuples containing (column name, classifier, input columns)
    """
    jtr = np.random.choice(np.arange(df.shape[0]), 1000, replace=True)
    jts = np.random.choice(np.arange(df.shape[0]), 100, replace=True)

    deterministic_num = []
    numerical_cols = [i for i in df.columns if df[i].dtype != "object"]
    if len(numerical_cols) <= 1:
        return deterministic_num, numerical_cols

    for j, i in enumerate(tqdm(numerical_cols)):
        clf = LinearRegression(n_jobs=-1, positive=True, fit_intercept=True).fit(
            df[numerical_cols].fillna(0).drop(i, axis=1).iloc[jtr],
            df[i].fillna(0).iloc[jtr],
        )

        score = clf.score(
            df[numerical_cols].fillna(0).drop(i, axis=1).iloc[jts],
            df[i].fillna(0).iloc[jts],
        )
        if score > 0.9999:
            numerical_cols.remove(i)
            deterministic_num.append([i, clf, copy.deepcopy(numerical_cols)])

    return deterministic_num


def find_ordinal_columns(df, cat_columns):
    """This test determines whether any of the columns
    containing nominal labels is characterised by ordinal properties.

    Args:
        df (_type_): _description_
        cat_columns (_type_): _description_

    Returns:
        _type_: _description_
    """
    ordinal_dicts = {}

    for i in tqdm(cat_columns):
        if 2 < df[i].nunique() < 255:
            s = []
            unique = df[i].fillna("N/A").unique()
            for j in unique:
                lp = np.array([int(i) for i in re.findall(r"\d+", j.replace(".", ""))])
                if len(lp) == 0:
                    lp = np.array([-1])

                s.append((lp.max() + lp.min()) / 2)

            possible_dict = dict(
                zip([unique[i] for i in np.argsort(s)], np.arange(unique.shape[0])),
            )

            if (
                np.unique(list(possible_dict.values())).shape[0]
                / len(list(possible_dict.values()))
                == 1
            ) and (np.mean(s) > 0):
                ordinal_dicts[i] = possible_dict

    return ordinal_dicts


def get_features_correlation(X):
    """_summary_

    Args:
        X (_type_): _description_
    """

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
    print("Calculating correlation matrix:")
    Xsample = copy.deepcopy(X)
    if X.shape[1] > 100:
        n = 500
    else:
        n = 2000
    samples = min(n, X.shape[0])
    Xsample = Xsample.sample(n=samples)
    for feature_i in tqdm(Xsample.columns):
        for feature_j in Xsample.columns:
            confusion_matrix = pd.crosstab(
                Xsample[feature_i].fillna(0),
                Xsample[feature_j].fillna(0),
            )

            features_correlation[feature_i].loc[feature_j] = float(
                round(_cramers_corrected_stat(confusion_matrix), 4),
            )

    return features_correlation


def get_label_correlation(
    Xproc,
    cat_cols,
    p_tr=0.75,
    delta_tr=0.05,
    n_min=100,
):
    """_summary_

    Args:
        Xproc (_type_): _description_
        cat_cols (_type_): _description_
        p_tr (float, optional): _description_. Defaults to 0.75.
        delta_tr (float, optional): _description_. Defaults to 0.05.
        n_min (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    list2d = list(cat_cols.values())
    merged = list(itertools.chain(*list2d))
    corr = []
    Xnp = Xproc.values
    for j in tqdm(range(len(merged))):
        Xt = Xnp[Xnp[:, j] == 1]
        d = Xt.shape[0]
        sample2 = np.random.choice(np.arange(d), d, replace=False)
        for i in range(len(merged)):
            p = Xt[:, i].sum() / d
            p2 = Xnp[sample2, i][sample2].sum() / d
            delta = p - p2
            if p > p_tr and i != j and d > n_min and delta > delta_tr:
                corr.append((merged[i], merged[j], p, d / Xproc.shape[0]))

    return corr


def column_a_greater_than_b(x, column_types, t=1.0):
    """_summary_

    Args:
        x (_type_): _description_
        column_types (_type_): _description_
        t (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    num_cols = [i for i in x.columns if column_types[i] == "number"]
    comp_matrix = dict()
    for i in tqdm(num_cols):
        for j in num_cols:
            d = (x[i] - x[j]).dropna()
            std = 1.0 * x[i].std()
            p = (d >= 0).sum() / d.shape[0]
            d_range = (np.abs(d) < std).sum() / d.shape[0]
            if i != j and p >= t and d_range >= t and d.shape[0] / x.shape[0] > 0.1:
                comp_matrix[i, j] = (d >= 0).sum() / d.shape[0]

    return comp_matrix
