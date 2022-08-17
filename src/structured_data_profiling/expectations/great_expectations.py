import numpy as np


def create_interval(string: str):
    bounds = string.split("_")[-1].replace("[", "").replace(")", "").split(",")
    return float(bounds[0]), float(bounds[1])


def add_column_expectations(batch, col_profiles):
    """
    Returns the sum of two decimal numbers in binary digits.

            Parameters:
                    batch (int): A decimal integer
                    column_types
                    col_profiles (int): Another decimal integer

            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    """
    for i in col_profiles:
        if col_profiles[i]['type'] == "number" and col_profiles[i]['na_frac'] < 1.0:

            batch.expect_column_values_to_be_between(
                i,
                min_value=col_profiles[i]["min"],
                max_value=col_profiles[i]["max"],
            )
            batch.expect_column_mean_to_be_between(
                i,
                col_profiles[i]["mean"] - col_profiles[i]["mean"] * 0.2,
                col_profiles[i]["mean"] + col_profiles[i]["mean"] * 0.2,
            )
            batch.expect_column_stdev_to_be_between(
                i,
                col_profiles[i]["std"] - col_profiles[i]["std"] * 0.3,
                col_profiles[i]["std"] + col_profiles[i]["std"] * 0.3,
            )
        elif col_profiles[i]['type'] == "string" \
                and col_profiles[i]['na_frac'] < 1.0 \
                and col_profiles[i]['n_unique'] > 1:
            batch.expect_column_most_common_value_to_be_in_set(
                i,
                col_profiles[i]["most_frequent"],
            )

    return batch


def add_conditional_expectations(batch, test_list, prepro, rare_labels):

    rl_dict = {}
    for i in rare_labels:
        rl_dict[i[0]] = list(i[1])

    for test in test_list:

        feat1 = [i for i in prepro.cat_cols.keys() if test[0] in prepro.cat_cols[i]][0]
        feat2 = [i for i in prepro.cat_cols.keys() if test[1] in prepro.cat_cols[i]][0]

        if str(batch[feat2].dtype) not in ["object", "bool"]:
            interval = create_interval(test[1])
            parse_arg = (
                    "`" + feat2
                    + "`>"
                    + str(interval[0])
                    + " and "
                    + "`" + feat2
                    + "`<"
                    + str(interval[1])
            )
        else:
            value2 = test[1].replace(feat2 + "_", "")
            parse_arg = "`" + feat2 + '`=="' + value2 + '"'

        if str(batch[feat1].dtype) not in ["object", "bool"]:
            interval = create_interval(test[0])
            batch.expect_column_values_to_be_between(
                column=feat1,
                min_value=interval[0],
                max_value=interval[1],
                condition_parser="pandas",
                row_condition=parse_arg,
                mostly=test[2],
            )
        else:
            value1 = test[0].replace(feat1 + "_", "")
            if value1 == "Grouped_labels":
                set1 = rl_dict[feat1]
            else:
                set1 = [value1]

            batch.expect_column_values_to_be_in_set(
                column=feat1,
                value_set=set1,
                condition_parser="pandas",
                row_condition=parse_arg,
                mostly=test[2],
            )

    return batch


def column_greater_than(data_batch, column_list):

    for test_i in column_list:
        data_batch.expect_column_pair_values_A_to_be_greater_than_B(column_A=test_i[0], column_B=test_i[1])

    return data_batch


    # TO DO
    # e2 = batch.expect_column_pair_values_A_to_be_greater_than_B('loan_amount', 'fico_average')
    # expect_column_pair_values_to_be_in_set
    # expect_column_values_to_be_dateutil_parseable
    # expect_column_values_to_be_null
    # expect_column_values_to_be_unique
    # expect_column_values_to_be_valid_date
    # expect_column_values_to_match_strftime_format
    # expect_multicolumn_sum_to_equal
