import great_expectations as ge


def create_interval(string: str):
    bounds = string.split('_')[-1].replace('[','').replace(')','').split(',')
    return float(bounds[0]), float(bounds[1])


def add_column_expectations(batch, dict):
    e1=batch.expect_column_values_to_be_between(i, min_value=dp.column_profiles[i]['min'], max_value=dp.column_profiles[i]['max'])
    e2= batch.expect_column_mean_to_be_between(i,dp.column_profiles[i]['mean']-dp.column_profiles[i]['mean']*0.2,dp.column_profiles[i]['mean']+dp.column_profiles[i]['mean']*0.2)
    e3= batch.expect_column_stdev_to_be_between(i,dp.column_profiles[i]['std']-dp.column_profiles[i]['std']*0.3,dp.column_profiles[i]['std']+dp.column_profiles[i]['std']*0.3)


    e2= batch.expect_column_most_common_value_to_be_in_set(i,['36 months'])
    e2 = batch.expect_column_pair_values_A_to_be_greater_than_B('loan_amount', 'fico_average')


def add_conditional_expectations(batch, test_list, prepro):

    test = test_list[0]

    feat1 = [i for i in prepro.cat_cols.keys() if test[0] in prepro.cat_cols[i]][0]
    feat2 = [i for i in prepro.cat_cols.keys() if test[1] in prepro.cat_cols[i]][0]
    # value2 = test[1].replace(feat2+'_', '')
    if str(batch[feat1].dtype) != 'object':
        interval = create_interval(test[0])
        set1 = batch[feat1][batch[feat1].between(interval[0], interval[1])].unique()

    else:
        value1 = test[0].replace(feat1 + '_', '')
        set1 = value1

    if str(batch[feat2].dtype) != 'object':
        interval = create_interval(test[0])
        parse_arg = feat2+'>'+str(interval[0])+' and '+feat2+'<'+str(interval[1])
    else:
        value2 = test[1].replace(feat2 + '_', '')
        parse_arg = feat2+'=="'+value2+'"'

    print(feat1,set1,parse_arg)
    return
    # batch.expect_column_values_to_be_in_set(
    #     column=feat1,
    #     value_set=set1, condition_parser='pandas',
    #     row_condition=parse_arg, mostly=0.99
    # )

    # expect_column_pair_values_a_to_be_greater_than_b
    # expect_column_pair_values_to_be_in_set
    # expect_column_values_to_be_dateutil_parseable
    # expect_column_values_to_be_null
    # expect_column_values_to_be_unique
    # expect_column_values_to_be_valid_date
    # expect_column_values_to_match_strftime_format
    # expect_multicolumn_sum_to_equal