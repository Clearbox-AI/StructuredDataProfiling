import dateparser
from functools import reduce
import numpy as np


def find_sequences(df, sequence_index, primary_key):

    sorted_seq = None
    if sequence_index is not None:
        if (df[sequence_index].isna()).sum() > 0:
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
            complete_ids = IDS
            # for i in range(len(dfs)):
            #     dfs[i].index = dfs[i][primary_key]
            #     dfs[i] = dfs[i].loc[IDS]
        else:
            # dfs = []
            # for i in sorted_seq:
            #     dfs.append(df[df[sequence_index] == i])
            complete_ids = None

    else:
        complete_ids = None