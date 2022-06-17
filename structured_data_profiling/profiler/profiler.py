from functools import reduce
import numpy as np
import pandas as pd
import dateparser
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import copy
import pickle
from structured_data_profiling.data_tests import *
from structured_data_profiling import Preprocessor


class DatasetProfiler:
    def __init__(self,
                 df_path: str,
                 primary_key: str = None,
                 sequence_index: str = None,
                 target: str = None,
                 n_samples: int = 10000,
                 regression: bool = False):

        df = pd.read_csv(df_path)

        if (primary_key is not None):
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
                self.data = [] #dfs
            else:
                # dfs = []
                # for i in sorted_seq:
                #     dfs.append(df[df[sequence_index] == i])

                self.data = [] #dfs
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
        self.warnings = None
        self.correlations = None
        self.bivariate_tests = None
        self.anomalies = None

        return

    def profile(self, tol: float = 1e-6):

        unique, binary, id_column, high_cardinality, rare_labels = check_cardinality(self.reduced_data_sample)
        self.column_profiles = {}
        self.warnings = {}

        self.high_cardinality = high_cardinality
        self.rare_labels = rare_labels
        self.unique_value = unique
        prepro = Preprocessor(self.reduced_data_sample)
        xp = prepro.transform(self.reduced_data_sample)
        self.bivariate_tests, self.anomalies = get_label_correlation(xp, prepro.cat_cols)

        self.column_profiler(self.reduced_data_sample)

        self.reduced_data_sample = self.reduced_data_sample.drop(list(unique.keys()), axis=1)
        for i in list(unique.keys()):
            self.warnings[i] = 'Unique value'

        self.reduced_data_sample = self.reduced_data_sample.drop(list(high_cardinality.keys()), axis=1)
        for i in list(high_cardinality.keys()):
            self.warnings[i] = 'High cardinality'

        self.correlations = get_features_correlation(self.reduced_data_sample)

        cat_columns = self.reduced_data_sample.columns[self.reduced_data_sample.dtypes == "object"]

        self.ordinal_columns = find_ordinal_columns(self.reduced_data_sample, cat_columns)

        for i in self.ordinal_columns.keys():
            for j in self.ordinal_columns[i].keys():
                self.reduced_data_sample[i] = self.reduced_data_sample[i].replace(j, self.ordinal_columns[i][j])

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

        return

    def column_profiler(self, data):

        for i in data.columns:

            loc_dict = {'dtype': data[i].dtype, 'nunique': data[i].nunique(),
                        'na_frac': data[i].isna().sum() / data.shape[0]}

            if data[i].dtype != 'object':
                loc_dict['non_negative'] = (data[i].fillna(0) >= 0).sum() / data.shape[0] == 1
                loc_dict['non_positive'] = (data[i].fillna(0) <= 0).sum() / data.shape[0] == 1
                loc_dict['min'] = data[i].min()
                loc_dict['max'] = data[i].max()
                loc_dict['mean'] = data[i].mean()
                loc_dict['representation'] = check_precision(data[i].sample(n=min(1000, data.shape[0])))
                if data[i].nunique() > 1:
                    loc_dict['distribution'] = fit_distributions(data[i].sample(n=min(1000, data.shape[0])))
                else:
                    loc_dict['distribution'] = 'unique value'
            else:
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

    def info(self):

        num_cols = [i for i in self.column_profiles if self.column_profiles[i]['dtype'] != object]
        cat_cols = [i for i in self.column_profiles if self.column_profiles[i]['dtype'] == object]

        print('Found ' + str(len(num_cols)) + ' numerical columns.')
        print('\n')
        print('Found ' + str(len(cat_cols)) + ' categorical columns.')
        print('\n')

        print('The following columns have been converted from categorical to numerical:')
        print('\n')

        for i in self.ordinal_columns:
            print(i, self.ordinal_columns[i])
            print('\n')

        l = [i for i in self.unique_value]
        print('Found ' + str(len(l)) + ' columns containing a unique value')
        print('\n')
        print(l)
        print('\n')

        l = [i[0] for i in self.deterministic_columns_regression]
        print('Found ' + str(len(l)) + ' deterministic numerical columns. ')
        print('\n')
        print(l)
        print('\n')

        l = [i[0] for i in self.deterministic_columns_binary]
        print('Found ' + str(len(l)) + ' deterministic binary columns. ')
        print('\n')
        print(l)
        print('\n')

        print('Warning, the following columns contain more than 75% of missing values:\n')
        print('\n')
        print([i for i in self.column_profiles if self.column_profiles[i]['na_frac'] > 0.75])
        print('\n')

        print('Warning, the following columns contain more than 99% of missing values:\n')
        print('\n')
        print([i for i in self.column_profiles if self.column_profiles[i]['na_frac'] > 0.99])
        print('\n')

        non_neg = [i for i in num_cols if self.column_profiles[i]['non_negative'] == True]
        non_pos = [i for i in num_cols if self.column_profiles[i]['non_positive'] == True]
        print('The following columns are non negative')
        print('\n')
        print(non_neg)
        print('\n')
        print('The following columns are non positive')
        print('\n')
        print(non_pos)

        return

    def save(self, name: str):

        for j, i in enumerate(self.data):
            self.data[j] = []

        with open(name, 'wb') as f:
            pickle.dump(self, f)


