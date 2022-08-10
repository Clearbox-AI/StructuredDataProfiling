import numpy as np
from collections import defaultdict


def lists_to_dict(a, b):
    l1 = len(np.unique(a))
    l2 = len(np.unique(b))
    new_dict = defaultdict(list)

    if l1 > l2:
        a1 = a
        b1 = b
        keys = [1, 0]
    else:
        a1 = b
        b1 = a
        keys = [0, 1]
    for j, i in enumerate(a1):
        new_dict[b1[j]].append(i)

    for i in new_dict.keys():
        new_dict[i] = np.unique(new_dict[i])

    return new_dict, keys


def reduce_dataframe(
    reduced_data_sample,
    column_profiles,
    high_cardinality_threshold=0.8,
):

    cat_cols = [
        i for i in column_profiles.keys() if column_profiles[i]["type"] == "string"
    ]
    unique = [
        i
        for i in column_profiles.keys()
        if column_profiles[i]["distribution"] == "unique_value"
    ]
    high_cardinality = [
        i
        for i in cat_cols
        if column_profiles[i]["n_unique_to_shape"] > high_cardinality_threshold
    ]

    reduced_data_sample = reduced_data_sample.drop(list(unique), axis=1)
    reduced_data_sample = reduced_data_sample.drop(list(high_cardinality), axis=1)
    too_much_info = [
        [i, column_profiles[i]["rare_labels (<5% frequency)"]]
        for i in cat_cols
        if len(column_profiles[i]["rare_labels (<5% frequency)"]) > 0 and i not in unique+high_cardinality
    ]

    for i in too_much_info:
        reduced_data_sample[i[0]] = reduced_data_sample[i[0]].replace(
            i[1],
            "Grouped_labels",
        )

    return reduced_data_sample
