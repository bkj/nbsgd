#!/usr/bin/env python

"""
    nbsgd.py
"""

from __future__ import print_function, division

import os
import re
import string
import numpy as np
from tqdm import tqdm

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import torch
from torch import nn
from torch.utils.data import dataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional as F

# --
# Helpers

def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def calc_r(y_i, x, y):
    x = x.sign()
    p = x[np.argwhere(y == y_i)[:,0]].sum(axis=0) + 1
    q = x[np.argwhere(y != y_i)[:,0]].sum(axis=0) + 1
    p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
    return np.log((p / p.sum()) / (q / q.sum()))

# --
# IO

X_train = np.load('./data/aclImdb/X_train.npy').item()
X_val = np.load('./data/aclImdb/X_val.npy').item()

X_train_words = np.load('./data/aclImdb/X_train_words.npy').item()
X_val_words = np.load('./data/aclImdb/X_val_words.npy').item()

y_train = np.load('./data/aclImdb/y_train.npy')
y_val = np.load('./data/aclImdb/y_val.npy')

train_dataset = dataset.TensorDataset(
    data_tensor=torch.from_numpy(X_train_words.toarray()).long(),
    target_tensor=torch.from_numpy(y_train).long()
)
val_dataset = dataset.TensorDataset(
    data_tensor=torch.from_numpy(X_val_words.toarray()).long(),
    target_tensor=torch.from_numpy(y_val).long(),
)


# --
# Define model

class DotProdNB(nn.Module):
    def __init__(self, vocab_size, n_classes, r, w_adj=0.4, r_adj=10, lr=0.02, weight_decay=1e-6,
        r_requires_grad=False, r_lr=0.0):
        
        super(DotProdNB, self).__init__()
        
        # Init w
        self.w = nn.Embedding(vocab_size + 1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1, 0.1)
        self.w.weight.data[0] = 0
        
        # Init r
        self.r = nn.Embedding(vocab_size + 1, n_classes)
        self.r.weight.data = torch.Tensor(np.concatenate([np.zeros((1, n_classes)), r])).cuda()
        self.r.weight.requires_grad = r_requires_grad
        
        self.w_adj = w_adj
        self.r_adj = r_adj
        
        
        if r_requires_grad:
            self.opt = torch.optim.Adam([
                {"params" : self.w.parameters(), "lr" : lr},
                {"params" : self.r.parameters(), "lr" : lr * r_lr},
            ], weight_decay=weight_decay)
        else:
            params = [p for p in self.parameters() if p.requires_grad]
            self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
    def forward(self, feat_idx):
        w = self.w(feat_idx) + self.w_adj
        r = self.r(feat_idx)
        
        x = (w * r).sum(dim=1)
        x =  x / self.r_adj # !! ??
        return x
    
    def step(self, x, y):
        output = self(x)
        self.opt.zero_grad()
        loss = F.cross_entropy(output, y)
        loss.backward()
        self.opt.step()
        return loss.data[0]


def do_eval(model):
    _ = model.eval()
    pred, act = [], []
    for x, y in val_dataloader:
        x = Variable(x.cuda(), volatile=True)
        pred.append(model(x))
        act.append(y)
        
    pred = to_numpy(torch.cat(pred)).argmax(axis=1)
    act = to_numpy(torch.cat(act))
    
    return (pred == act).mean()


# --
# Train

vocab_size = 200000
n_classes = int(y_train.max()) + 1
num_epochs = 4
batch_size = 256

_ = np.random.seed(123)
_ = torch.manual_seed(123)
_ = torch.cuda.manual_seed(123)

r = np.column_stack([calc_r(i, X_train, y_train) for i in range(n_classes)])
model = DotProdNB(vocab_size, n_classes, r).cuda()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

losses, evals = [], []
for _ in range(num_epochs):
    _ = model.train()
    for x, y in train_dataloader:
        x, y = Variable(x.cuda()), Variable(y.cuda())
        losses.append(model.step(x, y))
    
    evals.append(do_eval(model))
    print("acc=%f" % evals[-1])

final_acc = do_eval(model)
print("final_acc=%f" % final_acc)

