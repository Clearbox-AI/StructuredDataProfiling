import copy
from datetime import datetime
from functools import reduce

import dateparser
import numpy as np
import pandas as pd


def identify_dates(data, n_samples=10):
    data2 = copy.deepcopy(data)
    possible_dates = []
    a1 = datetime(2000, 1, 1).timestamp()
    a2 = datetime(datetime.now().year + 5, 1, 1).timestamp()
    timestamp = []
    timestamp_ms = []
    # check if format matches timestamp format
    for i in data2.columns:
        if str(data2[i].dtype) not in ["object", "category"]:
            time_column = data2[i].dropna()
            if (time_column > a1).sum() + (
                time_column < a2
            ).sum() > 1.9 * time_column.shape[0]:
                timestamp.append(i)

    for i in data2.columns:
        if str(data2[i].dtype) not in ["object", "category"]:
            time_column = data2[i].dropna() / 1000
            if (time_column > a1).sum() + (
                time_column < a2
            ).sum() > 1.9 * time_column.shape[0]:
                data2[i] = data2[i] / 1000
                timestamp_ms.append(i)
    # check if any int is compatible with dates

    cat_cols = data2.columns[data2.dtypes == "object"]
    for i in cat_cols:

        samples = np.random.choice(data2.shape[0], 1000)

        parsed_dates = [
            dateparser.parse(str(data2[i].iloc[j]), region="EU") is not None
            for j in samples
            if data2[i].isna().iloc[j] == False
        ]
        if len(parsed_dates) > 0:
            if sum(parsed_dates) / len(parsed_dates) > 0.99:
                possible_dates.append(i)

    dates = {}
    for i in possible_dates:
        dates[i] = "string/date"
    for i in timestamp:
        dates[i] = "number/timestamp"
    for i in timestamp_ms:
        dates[i] = "number/timestamp_ms"

    return dates


def sequence_profiles(df, sequence_index, primary_key):
    """
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    df (int): A decimal integer
                    sequence_index (int): Another decimal integer
                    primary_key
            Returns:
                    sorted_seq, complete_ids (str): Binary string of the sum of a and b
    """

    # X.groupby(['num_scontrino']).count().max(axis=1).value_counts().to_dict()

    sorted_seq = None
    if (df[sequence_index].isna()).sum() > 0:
        print("Initialisation failed, sequence_index contains NaNs")
        return
    else:
        seq_idxs = df[sequence_index].unique()
        seq_tmsp = []
        for i in seq_idxs:
            try:
                w = dateparser.parse(str(i), region="EU").timestamp()
            except Exception:
                w = None

            seq_tmsp.append(w)

        if None not in seq_tmsp:
            zipped_lists = zip(seq_tmsp, seq_idxs)
            sorted_zipped_lists = sorted(zipped_lists)

            sorted_seq = [element for _, element in sorted_zipped_lists]
        else:
            sorted_seq = seq_idxs.sort()

        print("Found following sequence indexes:", sorted_seq)
        complete_ids = []
        if primary_key:
            dfs = []
            ids = []
            for i in sorted_seq:
                dfs.append(df[df[sequence_index] == i])
                ids.append(df[primary_key][df[sequence_index] == i])

            IDS = reduce(np.intersect1d, (*ids,))
            complete_ids.append(IDS)
        else:
            complete_ids.append(None)

    return sorted_seq, complete_ids


def datetime_tests(df_in, dates):
    X = copy.deepcopy(df_in[dates])
    c = [(i, j) for i, j in zip(X.mean(axis=0).keys(), X.mean(axis=0).values)]
    c.sort(key=lambda t: t[1])
    for i in range(1, len(c)):
        X[i] = df_in[i].apply(lambda x: pd.Timestamp(x).timestamp())

    return c
