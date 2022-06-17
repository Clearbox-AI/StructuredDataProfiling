import great_expectations as ge


def add_expectations(batch, dict):
    e1=batch.expect_column_values_to_be_between(i, min_value=dp.column_profiles[i]['min'], max_value=dp.column_profiles[i]['max'])
    e2= batch.expect_column_mean_to_be_between(i,dp.column_profiles[i]['mean']-dp.column_profiles[i]['mean']*0.2,dp.column_profiles[i]['mean']+dp.column_profiles[i]['mean']*0.2)
    e3= batch.expect_column_stdev_to_be_between(i,dp.column_profiles[i]['std']-dp.column_profiles[i]['std']*0.3,dp.column_profiles[i]['std']+dp.column_profiles[i]['std']*0.3)


    e2= batch.expect_column_most_common_value_to_be_in_set(i,['36 months'])
    e2 = batch.expect_column_pair_values_A_to_be_greater_than_B('loan_amount', 'fico_average')

    batch.expect_column_values_to_be_in_set(
        column='sex',
        value_set=['Male'], condition_parser='pandas',
        row_condition='relationship=="Husband"', mostly=0.99
    )

    expect_column_pair_values_a_to_be_greater_than_b
    expect_column_pair_values_to_be_in_set
expect_column_values_to_be_dateutil_parseable
expect_column_values_to_be_null
expect_column_values_to_be_unique
expect_column_values_to_be_valid_date
expect_column_values_to_match_strftime_format
expect_multicolumn_sum_to_equal