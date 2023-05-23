import glob
import pandas as pd
import numpy as np
import pickle
import matplotlib
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm, tqdm_notebook
import tensorboardX as tbx
from collections import Counter
import uuid
from sklearn.metrics import classification_report, f1_score, roc_auc_score

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

import os


MIN_LEN = 11
MAX_LEN = 19


def load_dataset(filename, remove_duplicates = True):
    df = pd.read_csv(filename, sep="\t")
    df = df[(df["CDR3.sequence"].str.len() >= MIN_LEN).values & (df["CDR3.sequence"].str.len() <= MAX_LEN).values].reset_index(drop=True)
    if remove_duplicates:
        old_len = len(df)
        df.drop_duplicates("CDR3.sequence", inplace=True)
        print(" - Dropped", old_len - len(df), "duplicates")
        df.reset_index(drop=True, inplace=True)
    return df


def seq2vec(seq_list, vec_dict, max_len=MAX_LEN):
    # N, C, L
    if max_len <= 0:
        max_len = max([len(seq) for seq in seq_list])
    res = torch.zeros((len(seq_list), len(vec_dict["A"]), max_len))
    for seq_i, seq in enumerate(seq_list):
        for aa_i, aa in enumerate(seq.upper()):
            res[seq_i, :, aa_i] = vec_dict[aa]
    # Squeeze if last dimension is only one?
    return res


def load_dict(key = "kidera"):
    filename = "one_hot.pkl"
    if key == "onehot":
        filename = "one_hot.pkl"
    elif key == "kidera":
        filename = "kidera.pkl"
    elif key == "index":
        filename = "index.pkl"
    else:
        print("Unknown key:", key, " Returning the one-hot dictionary.")
    
    with open("features/" + filename, "rb") as file:
        return pickle.load(file)
    

def filter_specific(df_pos, df_neg):
    x = len(df_neg)
    df_neg = df_neg[~df_neg["CDR3.sequence"].isin(list(df_pos["CDR3.sequence"]))].reset_index(drop=True)
    print(" - Removed", x - len(df_neg), "sequences")
    return df_neg  


def create_batches(batch_neg, batch_pos):
    y_true = torch.tensor([0] * len(batch_neg), dtype=torch.float)
    tmp = batch_neg.abs().sum(dim=2)
    tmp[tmp > 1e-2] = 1
    
    yield (batch_neg, tmp.sum(dim=1)), y_true
    
    y_true = torch.tensor([1] * len(batch_pos), dtype=torch.float)
    tmp = batch_pos.abs().sum(dim=2)
    tmp[tmp > 1e-2] = 1
    
    yield (batch_pos, tmp.sum(dim=1)), y_true