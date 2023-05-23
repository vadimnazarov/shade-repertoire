import numpy as np
import pandas as pd
from utils import *


def load_data(data_tag):
    if data_tag == "initial-full-data":
        #
        # full data, 10k for test with seed 42
        #

        # Load the data and convert sequences to tensors
        #
        print("Loading and filtering data...")
        pos_path = "data/GSM3155092_P01_CRVstim_CD8_beta.txt.gz"
        neg_path = "data/GSM3155090_P01_unstim_CD8_beta.txt.gz"
        df_pos = load_dataset(pos_path)
        df_neg = load_dataset(neg_path)

        df_neg = filter_specific(df_pos, df_neg)
        print()

        print("Clonotype tables:")
        print(" - ", pos_path, ":", len(df_pos), "rows")
        print(" - ", neg_path, ":", len(df_neg), "rows")
        print()

        #
        # Convert sequences to tensors
        #
        test_size = 10000
        oh_dict = load_dict()

        seed = 42

        indices = np.random.choice(len(df_pos), test_size, replace=False)
        df_pos_test = df_pos.iloc[indices, :]
        df_pos_train = df_pos.drop(indices)

        indices = np.random.choice(len(df_neg), test_size, replace=False)
        df_neg_test = df_neg.iloc[indices, :]
        df_neg_train = df_neg.drop(indices)

        print("Train DFs sizes:", len(df_pos_train), len(df_neg_train))
        print("Test DFs sizes:", len(df_pos_test), len(df_neg_test))
        print()
    elif data_tag == "rc1-rc2-5k":
        #
        # pos > 1 read
        # neg > 2 read
        # 5k for test with seed 42
        #
        
        # Load the data and convert sequences to tensors
        #
        print("Loading and filtering data...")
        pos_path = "data/GSM3155092_P01_CRVstim_CD8_beta.txt.gz"
        neg_path = "data/GSM3155090_P01_unstim_CD8_beta.txt.gz"
        df_pos = load_dataset(pos_path)
        df_neg = load_dataset(neg_path)

        df_neg = filter_specific(df_pos, df_neg)
        print()

        print("Clonotype tables before filtering:")
        print(" - ", pos_path, ":", len(df_pos), "rows")
        print(" - ", neg_path, ":", len(df_neg), "rows")
        print()
        
        df_pos = df_pos.loc[df_pos["Read.count"] != 1, :].reset_index(drop=True)
        df_neg = df_neg.loc[df_neg["Read.count"] > 2, :].reset_index(drop=True)
        
        print("Clonotype tables after filtering:")
        print(" - ", pos_path, ":", len(df_pos), "rows")
        print(" - ", neg_path, ":", len(df_neg), "rows")
        print()

        #
        # Convert sequences to tensors
        #
        test_size = 5000
        oh_dict = load_dict()

        seed = 42

        indices = np.random.choice(len(df_pos), test_size, replace=False)
        df_pos_test = df_pos.iloc[indices, :]
        df_pos_train = df_pos.drop(indices)

        indices = np.random.choice(len(df_neg), test_size, replace=False)
        df_neg_test = df_neg.iloc[indices, :]
        df_neg_train = df_neg.drop(indices)

        print("Train DFs sizes:", len(df_pos_train), len(df_neg_train))
        print("Test DFs sizes:", len(df_pos_test), len(df_neg_test))
        print()
    else:
        raise ValueError('Unknown --data_tag value')
    
    return df_pos_test, df_pos_train, df_neg_test, df_neg_train, oh_dict