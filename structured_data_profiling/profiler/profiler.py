import copy
import pickle
from typing import List

import numpy as np
import pandas as pd
from pandas.core.tools.datetimes import _guess_datetime_format_for_array
from tqdm import tqdm

from structured_data_profiling.data_slicing import (
    check_column_balance,
    feature_importance,
    find_slices,
)
from structured_data_profiling.data_tests import (
    check_precision,
    column_a_greater_than_b,
    find_deterministic_columns_binary,
    find_deterministic_columns_regression,
    find_ordinal_columns,
    fit_distributions,
    get_features_correlation,
    get_label_correlation,
    identify_dates,
    is_text,
)
from structured_data_profiling.expectations import (
    add_column_expectations,
    add_conditional_expectations,
    column_greater_than,
)
from structured_data_profiling.preprocessor import Preprocessor
from structured_data_profiling.utils import reduce_dataframe


class DatasetProfiler:
    """

    The DatasetProfiler class .

    """

    def __init__(
        self,
        df_path: str,
        primary_key: str = None,
        sequence_index: str = None,
        target: str = None,
        regression: bool = False,
        protected_attributes: List = [],
        n_samples: int = None,
        compression: str = None,
        separator: str = ",",
        thousands: str = None,
        decimals: str = ".",
        encoding: str = None,
    ):
        """

        Parameters
        ----------
        df_path : str
            path of the CSV file to be profiled.
        primary_key : str, optional
            name of the column defining the CSV primary key (composite).
        sequence_index : str, optional
            name of the column from the CSV containing a sequence index.
        target : str, optional
            name of the column from the CSV containing a supervised learning target
            variable.
        compression : :obj:`int`, optional
            Description of `param3`.

        """
        df = pd.read_csv(
            df_path,
            compression=compression,
            sep=separator,
            decimal=decimals,
            thousands=thousands,
            encoding=encoding,
        )

        self.io_meta = {
            "compression": compression,
            "separator": separator,
            "thousands": thousands,
            "decimals": decimals,
            "encoding": encoding,
        }

        self.path = df_path

        if primary_key is not None:
            if type(primary_key) != list:
                primary_key = [primary_key]

            if max(df[primary_key].isna().sum()) == 0:
                self.primary_key = primary_key
            else:
                print("Error during initialisation: ")
                print(
                    "Error during initialisation: primary_key contains missing values",
                )
                return
        else:
            self.primary_key = None

        if target:
            self.target = target
            self.regression = regression
        else:
            self.target = None
            self.regression = False

        self.protected_attributes = protected_attributes
        contains_sequence = False

        if n_samples is None:
            n_samples = max(int(0.1 * df.shape[0]), 1000)

        samples = np.random.choice(
            df.shape[0],
            min(n_samples, df.shape[0]),
            replace=False,
        )
        self.n_samples = n_samples
        self.samples = samples

        self.data_sample = df.iloc[samples]

        self.original_shape = df.shape

        self.dataset_profile = {}

        self.reduced_data_sample = copy.deepcopy(self.data_sample)

        if self.primary_key:
            sequence_data = (
                self.reduced_data_sample.groupby(primary_key)
                .count()
                .max(axis=1)
                .value_counts()
            )
            if len(sequence_data.keys()) > 1:
                print("Identified sequential data.")
                contains_sequence = True
                self.sequence_index = sequence_index

        if contains_sequence is True:
            if self.sequence_index is not None:
                self.reduced_data_sample = self.reduced_data_sample.drop(
                    self.sequence_index,
                    axis=1,
                )
            self.dataset_profile["sequence_length"] = sequence_data

            self.reduced_data_sample = self.reduced_data_sample.groupby(
                primary_key,
            ).nth(0)

        if (self.primary_key is not None) and (contains_sequence is False):
            self.reduced_data_sample = self.reduced_data_sample.drop(
                self.primary_key,
                axis=1,
            )

        print("Identifying data types...")
        types = [
            "string" if i in ["object", "bool"] else "number"
            for i in self.reduced_data_sample.dtypes
        ]

        self.column_types = dict(zip(list(self.reduced_data_sample.columns), types))

        possible_dates = identify_dates(self.reduced_data_sample)
        for i in possible_dates.keys():
            self.column_types[i] = possible_dates[i]

        self.datetime_formats = {}
        string_dates = [
            i for i in self.column_types.keys() if self.column_types[i] == "string/date"
        ]

        for i in string_dates:
            try:
                format_time = _guess_datetime_format_for_array(
                    self.reduced_data_sample[i].sample(10).values,
                )
                self.datetime_formats[i] = format_time
            except Exception:
                self.datetime_formats[i] = "Could not infer datetime format"

        self.tests = None
        self.column_profiles = None

        self.prepro = None
        self.data_slices = None

        cat_columns = self.reduced_data_sample.columns[
            self.reduced_data_sample.dtypes == "object"
        ]

        ordinal_columns = find_ordinal_columns(
            self.reduced_data_sample,
            cat_columns,
        )

        self.ordinal_columns = ordinal_columns

        for i in cat_columns:
            if is_text(self.reduced_data_sample, i):
                self.column_types[i] = "text"

        if contains_sequence is True:
            self.sequence = True
            print("Dataset contains sequential data")
        else:
            self.sequence = False
        return

    def profile(self, tol: float = 1e-6, n_bins=5):

        self.column_profiler(self.reduced_data_sample, fit_distribution=False)

        self.reduced_data_sample = reduce_dataframe(
            self.reduced_data_sample,
            self.column_profiles,
        )
        self.dataset_profiler()
        # self.column_profiles = {}
        # self.high_cardinality = high_cardinality
        # self.rare_labels = rare_labels
        # self.unique_value = unique

        self.prepro = Preprocessor(column_types=self.column_types, n_bins=n_bins)
        self.tests = self.data_tests()

        try:
            self.slice_data()
        except Exception:
            print("Could not find data slices.")

        print("Profiling finished.")
        return

    def column_profiler(
        self,
        data,
        fit_distribution=True,
    ):

        self.column_profiles = {}
        print("Profiling columns:")
        for i in tqdm(data.columns):

            loc_dict = {
                "type": self.column_types[i],
                "n_unique": data[i].nunique(),
                "na_frac": data[i].isna().sum() / data.shape[0],
                "n_unique_to_shape": data[i].nunique() / data[i].shape[0],
                "distribution": "N/A",
            }

            if self.column_types[i] == "number":
                loc_dict["non_negative"] = (data[i].fillna(0) >= 0).sum() / data.shape[
                    0
                ] == 1
                loc_dict["non_positive"] = (data[i].fillna(0) <= 0).sum() / data.shape[
                    0
                ] == 1
                loc_dict["min"] = data[i].min()
                loc_dict["max"] = data[i].max()
                loc_dict["mean"] = data[i].mean()
                loc_dict["std"] = data[i].std()
                loc_dict["representation"] = check_precision(
                    data[i].sample(n=min(1000, data.shape[0])),
                )
                if loc_dict["n_unique"] == 2:
                    loc_dict["distribution"] = "Bernoulli"
                if loc_dict["n_unique"] <= 1:
                    loc_dict["distribution"] = "unique_value"

                if (loc_dict["n_unique"] > 2) and fit_distribution is True:
                    try:
                        loc_dict["distribution"] = fit_distributions(
                            data[i].sample(n=min(1000, data.shape[0])),
                        )
                    except Exception:
                        loc_dict[
                            "distribution"
                        ] = "Could not fit uni-variate distribution"

            elif self.column_types[i] == "string/date":

                if loc_dict["n_unique"] == 2:
                    loc_dict["distribution"] = "Bernoulli"
                    loc_dict["most_frequent"] = [
                        data[i].value_counts().keys()[0],
                        data[i].value_counts().iloc[0],
                    ]
                    loc_dict["least_frequent"] = [
                        data[i].value_counts().keys()[-1],
                        data[i].value_counts().iloc[-1],
                    ]
                elif loc_dict["n_unique"] <= 1:
                    loc_dict["distribution"] = "unique_value"
                    loc_dict["most_frequent"] = [
                        "N/A",
                        "N/A",
                    ]
                    loc_dict["least_frequent"] = [
                        "N/A",
                        "N/A",
                    ]
                else:
                    loc_dict["distribution"] = "N/A"
                    loc_dict["most_frequent"] = [
                        data[i].value_counts().keys()[0],
                        data[i].value_counts().iloc[0],
                    ]
                    loc_dict["least_frequent"] = [
                        data[i].value_counts().keys()[-1],
                        data[i].value_counts().iloc[-1],
                    ]

            elif self.column_types[i] in ["number/timestamp", "number/timestamp_ms"]:

                loc_dict["min"] = data[i].min()
                loc_dict["max"] = data[i].max()
                loc_dict["mean"] = data[i].mean()

                if loc_dict["n_unique"] == 2:
                    loc_dict["distribution"] = "Bernoulli"
                if loc_dict["n_unique"] <= 1:
                    loc_dict["distribution"] = "unique_value"

                loc_dict["distribution"] = "N/A"

            else:

                if loc_dict["n_unique"] == 2:
                    loc_dict["distribution"] = "Bernoulli"
                    loc_dict["most_frequent"] = [
                        data[i].value_counts().keys()[0],
                        data[i].value_counts().iloc[0],
                    ]
                    loc_dict["least_frequent"] = [
                        data[i].value_counts().keys()[-1],
                        data[i].value_counts().iloc[-1],
                    ]
                elif loc_dict["n_unique"] <= 1:
                    loc_dict["distribution"] = "unique_value"
                    loc_dict["most_frequent"] = [
                        "N/A",
                        "N/A",
                    ]
                    loc_dict["least_frequent"] = [
                        "N/A",
                        "N/A",
                    ]
                else:
                    loc_dict["distribution"] = "Categorical"
                    loc_dict["most_frequent"] = [
                        data[i].value_counts().keys()[0],
                        data[i].value_counts().iloc[0],
                    ]
                    loc_dict["least_frequent"] = [
                        data[i].value_counts().keys()[-1],
                        data[i].value_counts().iloc[-1],
                    ]
                counts = data[i].value_counts()
                loc_dict["rare_labels (<5% frequency)"] = list(
                    counts.keys()[(counts / data.shape[0]) < 0.05],
                )
                loc_dict["rare_labels (<1% frequency)"] = list(
                    counts.keys()[(counts / data.shape[0]) < 0.01],
                )

            self.column_profiles[i] = loc_dict

        return

    def dataset_profiler(self):

        duplicates = self.reduced_data_sample.duplicated()
        if len(duplicates > 0):
            duplicates_percentage = (
                self.reduced_data_sample[duplicates].shape[0]
                / self.reduced_data_sample.shape[0]
                * 100
            )
        else:
            duplicates_percentage = 0.0
        to_drop = [
            i
            for i in self.reduced_data_sample.columns
            if self.column_types[i] == "string/date"
        ]
        correlations = get_features_correlation(
            self.reduced_data_sample.drop(to_drop, axis=1),
        )
        self.dataset_profile["number_of_duplicates"] = duplicates_percentage
        self.dataset_profile["correlation_matrix"] = correlations

        return

    def summary(self):
        report = pd.DataFrame(columns=["profiling_type", "outcome"])
        num_cols = [
            i
            for i in self.column_profiles
            if self.column_profiles[i]["type"] == "number"
        ]
        """
        cat_cols = [
            i
            for i in self.column_profiles
            if self.column_profiles[i]["type"] == "string"
        ]
        """
        string_dates = [
            i for i in self.column_types.keys() if self.column_types[i] == "string/date"
        ]
        timestamps = [
            i
            for i in self.column_types.keys()
            if self.column_types[i]
            in [
                "number/timestamps",
                "number/timestamps_ms",
            ]
        ]
        report.loc[0] = [
            "column_types",
            {"numerical_columns": "num_cols", "categorical_columns": "cat_cols"},
        ]
        report.loc[1] = ["datetime_columns", string_dates + timestamps]
        report.loc[2] = [
            "duplicate_rows",
            self.dataset_profile["number_of_duplicates"],
        ]
        report.loc[3] = [
            "non_negative_colums",
            str(
                [
                    i
                    for i in num_cols
                    if self.column_profiles[i]["non_negative"] is True
                ],
            ),
        ]
        report.loc[4] = [
            "non_positive_colums",
            str(
                [
                    i
                    for i in num_cols
                    if self.column_profiles[i]["non_positive"] is True
                ],
            ),
        ]

        # dp.dataset_profile['correlation_matrix']
        cols = self.dataset_profile["correlation_matrix"].columns
        a = self.dataset_profile["correlation_matrix"].values - np.eye(
            self.dataset_profile["correlation_matrix"].shape[0],
        )
        indeces = np.triu_indices(self.dataset_profile["correlation_matrix"].shape[0])
        z = np.dstack((indeces[0], indeces[1]))

        list_corr = [(cols[i[0]], cols[i[1]]) for i in z[0] if a[i[0], i[1]] > 0.6]
        list_corr1 = [(cols[i[0]], cols[i[1]]) for i in z[0] if a[i[0], i[1]] > 0.6]

        del a

        report.loc[5] = ["highly correlated columns (>60%)", list_corr]
        report.loc[6] = ["highly correlated columns (>80%)", list_corr1]
        unique = [
            i
            for i in self.column_profiles.keys()
            if self.column_profiles[i]["distribution"] == "unique_value"
        ]
        report.loc[7] = ["Columns containing unique values", unique]

        return report

    def data_tests(self):

        data_tests = {}
        xp = self.prepro.transform(self.reduced_data_sample)
        xp_nan = self.prepro.transform_missing(self.reduced_data_sample)
        try:
            data_tests["is_greater_than"] = column_a_greater_than_b(
                self.reduced_data_sample,
                self.column_types,
                t=0.95,
            )
        except Exception:
            print("Could not complete is_greater_than tests.")
            data_tests["is_greater_than"] = None

        print("Finding bi-variate tests...")
        try:
            bivariate_tests = get_label_correlation(
                xp,
                self.prepro.cat_cols,
                p_tr=0.66,
                delta_tr=0.05,
            )
            data_tests["bivariate_tests"] = bivariate_tests

        except Exception:
            print("Could not complete bivariate tests.")
            data_tests["bivariate_tests"] = None

        print("Finding missing-values tests...")
        try:
            missing_values_tests = get_label_correlation(
                xp_nan,
                self.prepro.nan_cols,
                p_tr=0.66,
                delta_tr=0.05,
            )
            data_tests["missing_values_tests"] = missing_values_tests
        except Exception:
            print("Could not complete missing_values_tests tests.")
            data_tests["missing_values_tests"] = None

        """
        cat_columns = self.reduced_data_sample.columns[
            self.reduced_data_sample.dtypes == "object"
        ]
        """

        # for i in self.ordinal_columns.keys():
        #     for j in self.ordinal_columns[i].keys():
        #         self.reduced_data_sample[i] = self.reduced_data_sample[i].replace(j,
        #         self.ordinal_columns[i][j])
        samples = np.random.choice(
            self.reduced_data_sample.shape[0],
            min(self.n_samples, self.reduced_data_sample.shape[0]),
        )

        self.reduced_data_sample = self.reduced_data_sample.iloc[samples, :]

        column_search = [
            i
            for i in self.column_profiles.keys()
            if self.column_profiles[i]["distribution"] == "Bernoulli"
        ]
        #
        print("Finding threshold columns...")
        try:
            deterministic_columns_binary, num_cols = find_deterministic_columns_binary(
                self.reduced_data_sample,
                column_search,
            )
            data_tests["deterministic_columns_binary"] = deterministic_columns_binary
            to_drop = [i[0] for i in deterministic_columns_binary]
        except Exception:
            data_tests["deterministic_columns_binary"] = []
            to_drop = []

        print("Finding linear combinations...")

        self.reduced_data_sample = self.reduced_data_sample.drop(to_drop, axis=1)
        try:
            linear_combinations, num_cols = find_deterministic_columns_regression(
                self.reduced_data_sample,
            )
            data_tests["linear_combinations"] = linear_combinations
        except Exception:
            print("Could not complete linear_combinations tests.")
            data_tests["linear_combinations"] = None

        return data_tests

    def slice_data(self):

        text = [i for i in self.column_types.keys() if self.column_types[i] == "text"]
        X = copy.deepcopy(self.data_sample.drop(text, axis=1))
        if self.primary_key:
            X = X.drop(self.primary_key, axis=1)
        c1 = check_column_balance(X, target=self.target)
        c1 = [i[0] for i in c1]

        if self.target:
            c = feature_importance(X.fillna(0), self.target, self.regression)
            c = [i[0] for i in c]
            c1.remove(c[-1])
            list_columns = [c1[-1], c[-1]]
        else:
            list_columns = [c1[-2], c1[-1]]

        list_columns = list_columns + self.protected_attributes
        slices = find_slices(X, list_columns[-2:])
        self.data_slices = slices

    def generate_expectations(self, docs=True, suite_name=None):
        import great_expectations as ge

        if suite_name is None:
            suite_name = self.path.split("/")[-1]
        data_context = ge.data_context.DataContext()
        suite = data_context.create_expectation_suite(
            suite_name,
            overwrite_existing=True,
        )
        batch = ge.dataset.PandasDataset(
            self.data_sample.reset_index(),
            expectation_suite=suite,
        )

        cat_cols = [
            i
            for i in self.column_profiles.keys()
            if self.column_profiles[i]["type"] == "string"
        ]
        rare_labels = [
            [i, self.column_profiles[i]["rare_labels (<5% frequency)"]]
            for i in cat_cols
            if len(self.column_profiles[i]["rare_labels (<5% frequency)"]) > 0
        ]
        batch = add_column_expectations(batch, self.column_profiles)
        batch = add_conditional_expectations(
            batch,
            self.tests["bivariate_tests"],
            self.prepro,
            rare_labels,
        )

        batch = column_greater_than(batch, self.tests["is_greater_than"])

        suite = batch.get_expectation_suite()
        data_context.save_expectation_suite(suite)
        if docs is True:
            data_context.build_data_docs()
        return suite

    def save(self, name: str):

        # self.data_sample = []
        # self.reduced_data_sample = []

        with open(name, "wb") as f:
            pickle.dump(self, f)
