from functools import reduce
import great_expectations as ge
import numpy as np
import pandas as pd
import dateparser
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import copy
import pickle
from structured_data_profiling.preprocessor import Preprocessor
from structured_data_profiling.data_tests import *
from structured_data_profiling.expectations import *
from structured_data_profiling.utils import *


class DatasetProfiler:
    def __init__(self,
                 df_path: str,
                 primary_key: str = None,
                 sequence_index: str = None,
                 target: str = None,
                 n_samples: int = 10000,
                 regression: bool = False):

        df = pd.read_csv(df_path)
        self.path = df_path

        if primary_key is not None:
            self.id_columns = primary_key
        else:
            self.id_columns = None

        if target:
            self.target = target
            self.regression = regression
        else:
            self.target = None
            self.regression = False

        sorted_seq = None
        if (sequence_index is not None):
            if ((df[sequence_index].isna()).sum() > 0):
                print('Initialisation failed, sequence_index contains NaNs')
                return
            else:
                seq_idxs = df[sequence_index].unique()
                seq_tmsp = []
                for i in seq_idxs:
                    try:
                        w = dateparser.parse(str(i), region='EU').timestamp()
                    except:
                        w = None

                    seq_tmsp.append(w)

                if None not in seq_tmsp:
                    zipped_lists = zip(seq_tmsp, seq_idxs)
                    sorted_zipped_lists = sorted(zipped_lists)

                    sorted_seq = [element for _, element in sorted_zipped_lists]
                else:
                    sorted_seq = seq_idxs.sort()

            print('Found following sequence indeces:', sorted_seq)

            if primary_key:
                dfs = []
                ids = []
                for i in sorted_seq:
                    dfs.append(df[df[sequence_index] == i])
                    ids.append(df[primary_key][df[sequence_index] == i])

                IDS = reduce(np.intersect1d, (*ids,))
                self.complete_ids = IDS
                # for i in range(len(dfs)):
                #     dfs[i].index = dfs[i][primary_key]
                #     dfs[i] = dfs[i].loc[IDS]
                self.data = []  # dfs
            else:
                # dfs = []
                # for i in sorted_seq:
                #     dfs.append(df[df[sequence_index] == i])

                self.data = []  # dfs
                self.complete_ids = None

        else:
            self.data = df
            self.complete_ids = None

        samples = np.random.choice(df.shape[0], min(n_samples, df.shape[0]))
        self.data_sample = df.iloc[samples]
        self.original_shape = df.shape
        self.original_dtypes = df.dtypes

        self.reduced_data_sample = copy.deepcopy(self.data_sample)

        if self.id_columns is not None:
            self.reduced_data_sample = self.reduced_data_sample.drop(self.id_columns, axis=1)

        self.ordinal_columns = None
        self.deterministic_columns_binary = None
        self.deterministic_columns_regression = None
        self.dictionaries = None

        if sorted_seq:
            self.sequence_index = sequence_index
        else:
            self.sequence_index = None

        self.precision = None
        self.unique_value = None
        self.column_profiles = None
        self.high_cardinality = None
        self.rare_labels = None
        self.feature_selection = None
        self.n_samples = n_samples
        self.correlations = None
        self.bivariate_tests = None
        self.anomalies = None
        self.prepro = None
        self.duplicates_percentage = None
        self.possible_dates = None

        return

    def profile(self, tol: float = 1e-6):

        self.possible_dates = identify_dates(self.reduced_data_sample)
        unique, binary, id_column, high_cardinality, rare_labels = check_cardinality(self.reduced_data_sample)

        self.duplicates_percentage = self.reduced_data_sample[self.reduced_data_sample.duplicated() == True].shape[0] \
                                     / self.reduced_data_sample.shape[0] * 100
        self.reduced_data_sample = shrink_labels(self.reduced_data_sample, rare_labels)
        self.column_profiles = {}

        self.high_cardinality = high_cardinality
        self.rare_labels = rare_labels
        self.unique_value = unique
        self.prepro = Preprocessor(self.reduced_data_sample)
        xp = self.prepro.transform(self.reduced_data_sample)
        self.bivariate_tests, self.anomalies = get_label_correlation(xp, self.prepro.cat_cols, p_tr=0.66, delta_tr=0.05)

        self.column_profiler(self.reduced_data_sample)

        num_cols = [i for i in self.column_profiles if self.column_profiles[i]['dtype'] != object]
        cat_cols = [i for i in self.column_profiles if self.column_profiles[i]['dtype'] == object]
        print('Found ' + str(len(num_cols)) + ' numerical columns and ' + str(len(cat_cols)) + ' categorical columns.')
        print('\n')

        non_neg = [i for i in num_cols if self.column_profiles[i]['non_negative'] is True]
        non_pos = [i for i in num_cols if self.column_profiles[i]['non_positive'] is True]

        if len(non_neg) > 0:
            print('The following columns are non negative')
            print('\n')
            print(non_neg)
            print('\n')
        if len(non_pos) > 0:
            print('The following columns are non positive')
            print('\n')
            print(non_pos)

        self.reduced_data_sample = self.reduced_data_sample.drop(list(unique.keys()), axis=1)

        self.reduced_data_sample = self.reduced_data_sample.drop(list(high_cardinality.keys()), axis=1)

        self.correlations = get_features_correlation(self.reduced_data_sample)

        cat_columns = self.reduced_data_sample.columns[self.reduced_data_sample.dtypes == "object"]

        self.ordinal_columns = find_ordinal_columns(self.reduced_data_sample, cat_columns)

        # for i in self.ordinal_columns.keys():
        #     for j in self.ordinal_columns[i].keys():
        #         self.reduced_data_sample[i] = self.reduced_data_sample[i].replace(j, self.ordinal_columns[i][j])

        samples = np.random.choice(self.reduced_data_sample.shape[0], min(self.n_samples,
                                                                          self.reduced_data_sample.shape[0]))
        self.reduced_data_sample = self.reduced_data_sample.iloc[samples, :]

        column_search = np.intersect1d(
            list(self.reduced_data_sample.columns[self.reduced_data_sample.dtypes != 'object']), list(binary.keys()))

        self.deterministic_columns_binary, num_cols = find_deterministic_columns_binary(self.reduced_data_sample,
                                                                                        column_search)

        to_drop = [i[0] for i in self.deterministic_columns_binary]
        self.reduced_data_sample = self.reduced_data_sample.drop(to_drop, axis=1)
        self.deterministic_columns_regression, num_cols = find_deterministic_columns_regression(
            self.reduced_data_sample)
        to_drop = [i[0] for i in self.deterministic_columns_regression]
        self.reduced_data_sample = self.reduced_data_sample.drop(to_drop, axis=1)

        if self.target is not None:
            self.feat_selection()
        print('Profiling finished.')
        return

    def column_profiler(self, data):

        for i in data.columns:

            loc_dict = {'dtype': data[i].dtype, 'nunique': data[i].nunique(),
                        'na_frac': data[i].isna().sum() / data.shape[0]}

            if data[i].dtype != 'object':
                loc_dict['col_type'] = 'numerical'
                loc_dict['non_negative'] = (data[i].fillna(0) >= 0).sum() / data.shape[0] == 1
                loc_dict['non_positive'] = (data[i].fillna(0) <= 0).sum() / data.shape[0] == 1
                loc_dict['min'] = data[i].min()
                loc_dict['max'] = data[i].max()
                loc_dict['mean'] = data[i].mean()
                loc_dict['std'] = data[i].std()
                loc_dict['representation'] = check_precision(data[i].sample(n=min(1000, data.shape[0])))
                if data[i].nunique() > 1:
                    try:
                        loc_dict['distribution'] = fit_distributions(data[i].sample(n=min(1000, data.shape[0])))
                    except:
                        loc_dict['distribution'] = 'Could not fit uni-variate distribution'
                else:
                    loc_dict['distribution'] = 'unique value'
            else:
                loc_dict['col_type'] = 'categorical'
                loc_dict['most_frequent'] = [data[i].value_counts().keys()[0], data[i].value_counts()[0]]
                loc_dict['least_frequent'] = [data[i].value_counts().keys()[-1], data[i].value_counts()[-1]]
                if loc_dict['nunique'] == 2:
                    loc_dict['distribution'] = 'Bernoulli'
                else:
                    loc_dict['distribution'] = 'Categorical'

            self.column_profiles[i] = loc_dict

        return

    def feat_selection(self, frac=0.1):

        if self.target is None:
            print('no target column specified')
            return
        K = int(frac * self.reduced_data_sample.shape[1])
        num_cols = [i for i in self.reduced_data_sample.columns if self.reduced_data_sample[i].dtype != object]
        cat_cols = [i for i in self.reduced_data_sample.columns if self.reduced_data_sample[i].dtype == object]
        temp_df = pd.concat([self.reduced_data_sample[num_cols],
                             self.reduced_data_sample[cat_cols].astype('category').apply(lambda x: x.cat.codes)],
                            axis=1).fillna(0)

        if self.regression is True:
            feat_select = SelectKBest(mutual_info_regression, k=K).fit(temp_df.drop(self.target, axis=1),
                                                                       temp_df[self.target])
        else:
            feat_select = SelectKBest(mutual_info_classif, k=K).fit(temp_df.drop(self.target, axis=1),
                                                                    temp_df[self.target])

        self.feature_selection = list(feat_select.get_feature_names_out())
        if self.target not in self.feature_selection:
            self.feature_selection.append(self.target)
        if (self.sequence_index is not None) and (self.sequence_index not in self.feature_selection):
            self.feature_selection.append(self.sequence_index)

        return

    def warnings(self):

        if self.duplicates_percentage > 1.:
            print(str(self.duplicates_percentage) + ' % of the rows has at least one duplicate.')
            print('\n')

        high_corr = (self.correlations.values-np.eye(self.correlations.shape[0]) > 0.8).sum()
        if high_corr > 0:
            print(str(high_corr/2) + ' columns are highly correlated (>80%)')
            print('\n')

        if self.ordinal_columns is not None:
            print('The following columns might be categorical/ordinal:')
            print('\n')

            for i in self.ordinal_columns:
                print(i, self.ordinal_columns[i])
                print('\n')

        l = [i for i in self.unique_value]
        if len(l) > 0:
            print('Found ' + str(len(l)) + ' columns containing a unique value')
            print('\n')
            print(l)
            print('\n')

        if len(self.rare_labels) > 0:
            print('The following categorical labels are too rare (frequency<0.005%):')
            print('\n')
            for j in self.rare_labels:
                print(j)
            print('\n')

        l = [i[0] for i in self.deterministic_columns_regression]
        if len(l) > 0:
            print('Found ' + str(len(l)) + ' deterministic numerical columns. ')
            print('\n')
            print(l)
            print('\n')

        l = [i[0] for i in self.deterministic_columns_binary]
        if len(l) > 0:
            print('Found ' + str(len(l)) + ' deterministic binary columns. ')
            print('\n')
            print(l)
            print('\n')

        l = [i for i in self.column_profiles if self.column_profiles[i]['na_frac'] > 0.75]
        if len(l) > 0:
            print('Warning, the following columns contain more than 75% of missing values:\n')
            print('\n')
            print(l)
            print('\n')

        l = [i for i in self.column_profiles if self.column_profiles[i]['na_frac'] > 0.99]
        if len(l) > 0:
            print('Warning, the following columns contain more than 99% of missing values:\n')
            print('\n')
            print(l)
            print('\n')

        return

    def generate_expectations(self, docs=True):

        data_context = ge.data_context.DataContext()
        suite = data_context.create_expectation_suite('local_suite', overwrite_existing=True)
        batch = ge.dataset.PandasDataset(self.data_sample, expectation_suite=suite)

        batch = add_column_expectations(batch, self.column_profiles)
        batch = add_conditional_expectations(batch, self.bivariate_tests, self.prepro, self.rare_labels)

        suite = batch.get_expectation_suite()
        data_context.save_expectation_suite(suite)
        if docs is True:
            data_context.build_data_docs()
        return suite

    def save(self, name: str):

        for j, i in enumerate(self.data):
            self.data[j] = []

        with open(name, 'wb') as f:
            pickle.dump(self, f)
