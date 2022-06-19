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


def shrink_labels(reduced_df, too_much_info):
    if len(too_much_info) > 0:
        columns_to_shrink_str = [i[0] for i in too_much_info if i[1].dtype == 'object']
        cats_to_shrink_str = [i[1] for i in too_much_info if i[1].dtype == 'object']
        columns_to_shrink_int = [i[0] for i in too_much_info if i[1].dtype != 'object']
        cats_to_shrink_int = [i[1] for i in too_much_info if i[1].dtype != 'object']
        to_shrink_str = dict(zip(columns_to_shrink_str, cats_to_shrink_str))
        to_shrink_int = dict(zip(columns_to_shrink_int, cats_to_shrink_int))

        reduced_df = reduced_df.replace(to_shrink_str, 'Grouped_labels')
        reduced_df = reduced_df.replace(to_shrink_int, -999999)

        return reduced_df