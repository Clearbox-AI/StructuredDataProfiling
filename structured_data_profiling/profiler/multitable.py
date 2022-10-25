import pickle
from typing import Dict

import pandas as pd

from .profiler import DatasetProfiler


class MultiTableProfiler:
    """

    The DatasetProfiler class .

    """

    def __init__(
        self,
        relational_metadata: Dict,
        target: Dict = None,
        n_samples: int = 10000,
        compression: str = None,
        separator: str = ",",
        thousands: str = None,
        decimals: str = ".",
        encoding: str = None,
    ):
        """

        Parameters
        ----------s
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
        self.profiles = []
        self.relational_metadata = relational_metadata
        self.io_meta = {
            "compression": compression,
            "separator": separator,
            "thousands": thousands,
            "decimals": decimals,
            "encoding": encoding,
        }

        tables = {}
        try:
            for i in relational_metadata.keys():
                # print(i, len(relational_metadata[i]))
                table_i = pd.read_csv(
                    relational_metadata[i][0],
                    compression=compression,
                    sep=separator,
                    decimal=decimals,
                    thousands=thousands,
                )

                table_i.index = table_i[relational_metadata[i][1]]
                table_i = table_i.drop(relational_metadata[i][1], axis=1)
                tables[i] = table_i
            # check if primary key exists
            # assign primary key to index
            # check if parent table exists
            #
        except FileNotFoundError:
            print("Could not read tables provided as paths")
            return

        completed = []
        to_process = list(relational_metadata.keys())
        w = 0
        while len(to_process) > 0 and w < 100:

            for i in to_process:
                len_meta = len(relational_metadata[i]) / 2
                if len_meta == 1:
                    completed.append(i)
                    to_process.remove(i)
                else:
                    foreign_dfs = [
                        relational_metadata[i][2 + 2 * j]
                        for j in range(int(len_meta) - 1)
                    ]
                    if all(elem in completed for elem in foreign_dfs):
                        completed.append(i)
                        to_process.remove(i)
            w += 1

        if w > 99:
            print("Could not build relational DAG, please check foreign keys provided")
            return

        metadata_merging = {}
        foreign_keys = {}

        for i in completed:
            len_meta = len(relational_metadata[i]) / 2
            if len_meta > 1:
                df = tables[i]
                columns = list(df.columns)
                for j in range(int(len_meta) - 1):
                    df = df.join(
                        tables[relational_metadata[i][2 + 2 * j]],
                        # lsuffix='_l_'+i,
                        rsuffix="_r_" + relational_metadata[i][2 + 2 * j],
                        on=relational_metadata[i][3 + 2 * j],
                    )
                    df = df.drop(relational_metadata[i][3 + 2 * j], axis=1)

                tables[i] = df.sample(min(n_samples, df.shape[0]))
                metadata_merging[i] = columns
                # foreign_keys[i] = relational_metadata[i][3 + 2 * i]
                # print(i, df.shape)
            else:
                metadata_merging[i] = []

        self.tables = tables
        self.metadata = metadata_merging
        self.foreign_keys = foreign_keys

        return

    def profile(self, n_samples=10000):

        for i in self.tables.keys():
            self.tables[i].to_csv("temp.csv", index_label="index")
            dp = DatasetProfiler(
                "temp.csv",
                primary_key="index",
                # regression=self.io_meta['regression'],
                n_samples=n_samples,
                # compression=self.io_meta['compression'],
                # separator=self.io_meta['separator'],
                # decimals=self.io_meta['decimals'],
                # thousands=self.io_meta['thousands'],
                # encoding=self.io_meta['encoding']
            )

            dp.profile()
            # dp.generate_expectations(suite_name=str(i))

            dp.data_sample = []
            dp.reduced_data_sample = []

            self.profiles.append(dp)

        print("Profiling finished.")

        return

    def save(self, name: str):

        with open(name, "wb") as f:
            pickle.dump(self, f)
