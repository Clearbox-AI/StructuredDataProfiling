import great_expectations as ge


def create_interval(string: str):
    bounds = string.split('_')[-1].replace('[','').replace(')','').split(',')
    return float(bounds[0]), float(bounds[1])


def add_column_expectations(batch, col_profiles):

    for i,j in enumerate(col_profiles):
        e1= batch.expect_column_values_to_be_between(i, min_value=col_profiles[i]['min'], max_value=col_profiles[i]['max'])
        e2= batch.expect_column_mean_to_be_between(i,col_profiles[i]['mean']-col_profiles[i]['mean']*0.2,col_profiles[i]['mean']+col_profiles[i]['mean']*0.2)
        e3= batch.expect_column_stdev_to_be_between(i,col_profiles[i]['std']-col_profiles[i]['std']*0.3,col_profiles[i]['std']+col_profiles[i]['std']*0.3)


    e2= batch.expect_column_most_common_value_to_be_in_set(i,['36 months'])


def add_conditional_expectations(batch, test_list, prepro):

    for test in test_list:

        feat1 = [i for i in prepro.cat_cols.keys() if test[0] in prepro.cat_cols[i]][0]
        feat2 = [i for i in prepro.cat_cols.keys() if test[1] in prepro.cat_cols[i]][0]

        if str(batch[feat1].dtype) != 'object':
            interval = create_interval(test[0])
            set1 = batch[feat1][batch[feat1].between(interval[0], interval[1])].unique()

        else:
            value1 = test[0].replace(feat1 + '_', '')
            set1 = value1

        if str(batch[feat2].dtype) != 'object':
            interval = create_interval(test[1])
            parse_arg = feat2+'>'+str(interval[0])+' and '+feat2+'<'+str(interval[1])
        else:
            value2 = test[1].replace(feat2 + '_', '')
            parse_arg = feat2+'=="'+value2+'"'

        print(feat1,set1,parse_arg,test[2])

        batch.expect_column_values_to_be_in_set(
            column=feat1,
            value_set=set1,
            condition_parser='pandas',
            row_condition=parse_arg,
            mostly=test[2]
        )
    return


    # TO DO
    #e2 = batch.expect_column_pair_values_A_to_be_greater_than_B('loan_amount', 'fico_average')

    # expect_column_pair_values_a_to_be_greater_than_b
    # expect_column_pair_values_to_be_in_set
    # expect_column_values_to_be_dateutil_parseable
    # expect_column_values_to_be_null
    # expect_column_values_to_be_unique
    # expect_column_values_to_be_valid_date
    # expect_column_values_to_match_strftime_format
    # expect_multicolumn_sum_to_equal