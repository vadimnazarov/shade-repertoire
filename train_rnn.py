from argparse import ArgumentParser

import pandas as pd
import numpy as np
import editdistance as ed
from tqdm import tqdm
from collections import Counter
import time
import json
import sys

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from tensorboardX import SummaryWriter

from sklearn.metrics import accuracy_score, f1_score

import uuid

import utils
import importlib
importlib.reload(utils)
from utils import *
from data import *


class CMVModel(nn.Module):
    
    def __init__(self, input_size, hidden_size=32, num_layers=1, rnn_dropout=0, bn=False, concat=False, attn=True):
        super(CMVModel, self).__init__()
        
        self._hidden_size = hidden_size
        self._bn = bn
        self._concat = concat
        self._attn = attn
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=self._hidden_size, num_layers=num_layers, bidirectional=True, dropout=rnn_dropout, batch_first=True)
        
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
        
        if self._bn:
            layers = [nn.BatchNorm1d(self._hidden_size*(self._concat+1)),
                      nn.LeakyReLU(),
                      nn.Linear(self._hidden_size*(self._concat+1), 1), 
                      nn.Sigmoid()]
                
#             layers = [nn.BatchNorm1d(self._hidden_size*(self._concat+1)),
#                       nn.LeakyReLU(),
#                       nn.Linear(self._hidden_size*(self._concat+1), self._hidden_size*(self._concat+1)),
#                       nn.BatchNorm1d(self._hidden_size*(self._concat+1)),
#                       nn.LeakyReLU(),
#                       nn.Linear(self._hidden_size*(self._concat+1), 1), 
#                       nn.Sigmoid()]
        
            nn.init.kaiming_uniform_(layers[-2].weight)
        else:
            layers = [nn.LeakyReLU(),
                      nn.Linear(self._hidden_size*(self._concat+1), 1), 
                      nn.Sigmoid()]
            nn.init.kaiming_uniform_(layers[-2].weight)
            
#             layers = [nn.LeakyReLU(),
#                       nn.Linear(self._hidden_size*(self._concat+1), self._hidden_size*(self._concat+1)),
#                       nn.LeakyReLU(),
#                       nn.Linear(self._hidden_size*(self._concat+1), 1), 
#                       nn.Sigmoid()]
#             nn.init.kaiming_uniform_(layers[1].weight)
#             nn.init.kaiming_uniform_(layers[-2].weight)
        
        self.final = nn.Sequential(*layers)
        
        if self._attn:
            self.attn = nn.Sequential(nn.Linear(self._hidden_size*2, 1), 
                                      nn.Tanh())
            nn.init.kaiming_uniform_(self.attn[0].weight)
        
    
    def forward(self, batch, lens_vec):
        x = pack_padded_sequence(batch, lens_vec, batch_first=True, enforce_sorted=False)
        
        if not self._attn:
#             x = self.rnn(x)[1]
#             x = self.rnn(batch)[1]
    
            if self._concat:
                x = torch.cat((x[-2, :, :], x[-1, :, :]), dim=1)
            else:
                x = (x[-2, :, :] + x[-1, :, :]).mul(.5)
        else:
            x = self.rnn(x)[0]
#             x = self.rnn(batch)[0]
            x, lens_vec = pad_packed_sequence(x, batch_first=True)
            # if dropout_cl:
#                 indices = torch.randperm(x.shape[0])[:int((1. - dropout_cl) * x.shape[0])]
#                 x, lens_vec = x[indices], lens_vec[indices]

            # implement masking here
            maxlen = lens_vec.max()
            mask = torch.arange(lens_vec.max())[None, :] < lens_vec[:, None]
            mask = mask.to(x.get_device())
        
            attn_weights = self.attn(x).masked_fill(mask.unsqueeze(2) == False, -1e10)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            x = x * attn_weights
            x = x.sum(1)

        x = self.final(x)

        return x
    
    
def run_epoch(model, criterion, optimiser, dl, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()
        
    loss_list = []
    pred_list = []
    true_list = []
    
    start = time.time()
    for batch_neg, batch_pos in dl:
        optimiser.zero_grad()
        loss = 0
        
        for (batch, lens), y_true in create_batches(batch_neg, batch_pos):
#             batch = pack_padded_sequence(batch.to("cuda"), lens, batch_first=True, enforce_sorted=False)
#             y_pred = model(batch).reshape((-1,))
            y_pred = model(batch.to("cuda"), lens).reshape((-1,))
            loss += criterion(y_pred, y_true.to("cuda"))
            
            y_pred_round = y_pred.cpu().detach().round().numpy()
            pred_list.append(y_pred_round)
            true_list.append(y_true.cpu().detach().numpy())
        
        if train_mode:
            loss.backward()
            optimiser.step()
        
        loss_list.append(loss.cpu().item())
    end = time.time()
    
    return loss_list, pred_list, true_list, end - start


if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int, 
                        default=64)
    parser.add_argument("--epochs", "-e", type=int, 
                        default=15, help="Epochs per one mega-epoch")
#     parser.add_argument("--mega_epochs", "--me", type=int,
#                         default=30, help="Number of mega-epochs")
    parser.add_argument("--learning_rate", "--lr", type=float, 
                        default=0.001)
    parser.add_argument("--weight_decay", "--wd", type=float, 
                        default=1e-2)
    parser.add_argument("--data_tag", "--dt", type=str, 
                        default="initial-full-data")
    
    parser.add_argument("--hidden_size", type=int, 
                        default=64)
    parser.add_argument("--num_layers", type=int, 
                        default=2)
    parser.add_argument("--concat", type=bool, 
                        default=True)
    parser.add_argument("--attn", type=bool, 
                        default=True)
    parser.add_argument("--dropout", type=float, 
                        default=0.0)
    parser.add_argument("--bn", type=bool, 
                        default=False)
    
    parser.add_argument("--logs", type=bool, 
                        default=True)
    
    args = parser.parse_args()
    assert(args.epochs > 0)
    args = vars(args)
#     assert(args.mega_epochs > 0)

    torch.backends.cudnn.benchmark = True
    
    #
    # Load the data and convert sequences to tensors
    #
    df_pos_test, df_pos_train, df_neg_test, df_neg_train, oh_dict = load_data(args["data_tag"])
    
    # Sanity check
#     print(set(df_pos_test["CDR3.sequence"]).intersection(set(df_pos_train["CDR3.sequence"])))
#     print(set(df_pos_test["CDR3.sequence"]).intersection(set(df_neg_test["CDR3.sequence"])))
#     print(set(df_neg_test["CDR3.sequence"]).intersection(set(df_neg_train["CDR3.sequence"])))
#     print(set(df_pos_train["CDR3.sequence"]).intersection(set(df_neg_train["CDR3.sequence"])))

    print("Converting sequences to tensors...", end="\t")
    X_pos_train = seq2vec(list(df_pos_train["CDR3.sequence"]), oh_dict)
    X_pos_test = seq2vec(list(df_pos_test["CDR3.sequence"]), oh_dict)
    X_neg_train = seq2vec(list(df_neg_train["CDR3.sequence"]), oh_dict)
    X_neg_test = seq2vec(list(df_neg_test["CDR3.sequence"]), oh_dict)
    print("Done!")

    print("Train Tensors sizes:", len(X_pos_train), len(X_neg_train))
    print("Test Tensors sizes:", len(X_pos_test), len(X_neg_test))
    print()
    
    print(X_pos_train.shape, X_pos_test.shape)
    print(X_neg_train.shape, X_neg_test.shape)
    print()
    
    #
    # Prepare the model and data loaders
    #
    
    model = CMVModel(len(oh_dict["A"]), hidden_size=args["hidden_size"], num_layers=args["num_layers"], rnn_dropout=args["dropout"], bn=args["bn"], concat=args["concat"], attn=args["attn"]).to("cuda")
    print(model)
    print()
    
    criterion = nn.BCELoss()
    
    sampler_neg = RandomSampler(X_neg_train.transpose(2, 1), replacement=False, num_samples=None)
    sampler_pos = RandomSampler(X_pos_train.transpose(2, 1), replacement=True, num_samples=len(X_neg_train))
    
    dl_train_neg = DataLoader(X_neg_train.transpose(2, 1), sampler=sampler_neg, batch_size=args["batch_size"], num_workers=1, pin_memory=True, drop_last=True)
    dl_train_pos = DataLoader(X_pos_train.transpose(2, 1), sampler=sampler_pos, batch_size=args["batch_size"], num_workers=1, pin_memory=True, drop_last=True)
    dl_test = DataLoader(TensorDataset(X_neg_test.transpose(2, 1), X_pos_test.transpose(2, 1)), batch_size=2048, shuffle=False, num_workers=1, pin_memory=True, drop_last=False)

    #
    # Train
    #
    tag = str(uuid.uuid4())[:4]
    
    if args["logs"]:
        writer = SummaryWriter("./experiments/logs/" + tag)
        print("Writing to ./experiments with ID '", tag, "'", sep="")
        with open("./experiments/parameters/" + tag + ".txt", "w") as par_file:
            command_string = " ".join(sys.argv)
            d = args
            d["command"] = command_string
            par_file.write(json.dumps(d, sort_keys=True))
    
    step = 1
    
    learning_rate = args["learning_rate"]
    for mega_epoch in range(2):
        print(learning_rate)
        optimiser = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args["weight_decay"])
        
        for i_epoch in range(mega_epoch*args["epochs"] + 1, (mega_epoch + 1) * args["epochs"] + 1):
            loss_list, pred_list, true_list, exec_time = run_epoch(model, criterion, optimiser, zip(dl_train_neg, dl_train_pos), True)
            pred_list = np.concatenate(pred_list).ravel()
            true_list = np.concatenate(true_list).ravel()
            cr = classification_report(true_list, pred_list, target_names=["CMV-", "CMV+"], output_dict=False)

            print("======================================")
            print("Epoch:{0:3}".format(i_epoch))
            print("  - Train (in {0:3.2f}s), loss {1:6.5f}".format(exec_time, np.mean(loss_list)), )
            print(cr[:162])

            if args["logs"]:
                cr = classification_report(true_list, pred_list, target_names=["CMV-", "CMV+"], output_dict=True)
                writer.add_scalar("train/loss", np.mean(loss_list), step)
                writer.add_scalar("train/f1", cr["CMV+"]["f1-score"], step)
                writer.add_scalar("train/precision", cr["CMV+"]["precision"], step)
                writer.add_scalar("train/recall", cr["CMV+"]["recall"], step)

            loss_list, pred_list, true_list, exec_time = run_epoch(model, criterion, optimiser, dl_test, False)
            pred_list = np.concatenate(pred_list).ravel()
            true_list = np.concatenate(true_list).ravel()
            cr = classification_report(true_list, pred_list, target_names=["CMV-", "CMV+"], output_dict=False)
            print("  - Test (in {0:3.2f}s), loss {1:6.5f}".format(exec_time, np.mean(loss_list)), )
            print(cr[:162])
            print("======================================")
            print()

            if args["logs"]:
                cr = classification_report(true_list, pred_list, target_names=["CMV-", "CMV+"], output_dict=True)
                writer.add_scalar("val/loss", np.mean(loss_list), step)
                writer.add_scalar("val/f1", cr["CMV+"]["f1-score"], step)
                writer.add_scalar("val/precision", cr["CMV+"]["precision"], step)
                writer.add_scalar("val/recall", cr["CMV+"]["recall"], step)
                torch.save(model.state_dict(), "./experiments/models/" + tag + ".pth")

            step += 1
            
        learning_rate /= 10