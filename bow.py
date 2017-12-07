#!/usr/bin/env python

"""
    bow.py
"""

from __future__ import print_function, division

import os
import re
import string
import numpy as np
from tqdm import tqdm

from rsub import *
from matplotlib import pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import torch
from torch import nn
from torch.utils.data import dataset, DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data.sampler import SequentialSampler, RandomSampler, BatchSampler


# --
# Helpers

def texts_from_folders(src, names):
    texts,labels = [],[]
    for idx,name in enumerate(names):
        path = os.path.join(src, name)
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(open(fpath).read())
            labels.append(idx)
    return texts,np.array(labels)


def bow2adjlist(X, maxcols=None):
    x = coo_matrix(X)
    _, counts = np.unique(x.row, return_counts=True)
    pos = np.hstack([np.arange(c) for c in counts])
    adjlist = csr_matrix((x.col + 1, (x.row, pos)))
    datlist = csr_matrix((x.data, (x.row, pos)))
    
    if maxcols is not None:
        adjlist, datlist = adjlist[:,:maxcols], datlist[:,:maxcols]
    
    return adjlist, datlist


def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def one_hot(a, c):
    return np.eye(c)[a]


def calc_r(y_i, x, y):
    x = x.sign()
    p = x[np.argwhere(y == y_i)[:,0]].sum(axis=0) + 1
    q = x[np.argwhere(y != y_i)[:,0]].sum(axis=0) + 1
    p, q = np.asarray(p).squeeze(), np.asarray(q).squeeze()
    return np.log((p / p.sum()) / (q / q.sum()))


# --
# IO

names = ['neg', 'pos']
text_train, y_train = texts_from_folders('data/aclImdb/train', names)
text_val, y_val = texts_from_folders('data/aclImdb/test', names)

# --
# Preprocess

max_features = 200000
max_len = 1000

re_tok = re.compile('([%s“”¨«»®´·º½¾¿¡§£₤‘’])' % string.punctuation)
tokenizer = lambda x: re_tok.sub(r' \1 ', x).split()

vectorizer = CountVectorizer(
    ngram_range=(1,3),
    tokenizer=tokenizer, 
    max_features=max_features
)
X_train = vectorizer.fit_transform(text_train)
X_val = vectorizer.transform(text_val)

vocab_size = X_train.shape[1]
n_classes = int(y_train.max()) + 1

X_train_words, _ = bow2adjlist(X_train, maxcols=1000)
X_val_words, _ = bow2adjlist(X_val, maxcols=1000)

train_dataset = dataset.TensorDataset(
    data_tensor=torch.from_numpy(X_train_words.toarray()).long(),
    target_tensor=torch.from_numpy(one_hot(y_train, n_classes).astype(np.float32)),
)
val_dataset = dataset.TensorDataset(
    data_tensor=torch.from_numpy(X_val_words.toarray()).long(),
    target_tensor=torch.from_numpy(one_hot(y_val, n_classes).astype(np.float32)),
)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# --
# Define model

class DotProdNB(nn.Module):
    def __init__(self, vocab_size, n_classes, r, w_adj=0.4, r_adj=10, 
        lr=0.02, weight_decay=1e-6):
        
        super(DotProdNB, self).__init__()
        
        # Init w
        self.w = nn.Embedding(vocab_size + 1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1,0.1)
        
        # Init r
        self.r = nn.Embedding(vocab_size + 1, n_classes)
        self.r.weight.data = torch.Tensor(np.concatenate([np.zeros((1, n_classes)), r])).cuda()
        self.r.weight.requires_grad = False
        
        self.w_adj = w_adj
        self.r_adj = r_adj
        
        params = [p for p in self.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
    def forward(self, feat_idx):
        w = self.w(feat_idx)
        r = self.r(feat_idx)
        
        x = ((w + self.w_adj) * r / self.r_adj).sum(dim=1)
        return F.softmax(x)
    
    def step(self, x, y):
        output = self(x)
        self.opt.zero_grad()
        loss = F.cross_entropy(output, y)
        loss.backward()
        self.opt.step()
        return loss.data[0]


r = np.column_stack([calc_r(i, X_train, y_train) for i in range(n_classes)])
model = DotProdNB(vocab_size, n_classes, r).cuda()

# --
# Train

losses, evals = [], []
for _ in range(4):
    _ = model.train()
    for x, y in train_dataloader:
        y = y.max(dim=1)[1].long()
        x, y = Variable(x.cuda()), Variable(y.cuda())
        losses.append(model.step(x, y))
    
    evals.append(do_eval())
    print(evals[-1])

_ = plt.plot(losses)
show_plot()

# --
# Eval

def do_eval():
    _ = model.eval()
    pred, act = [], []
    for x, y in val_dataloader:
        x = Variable(x.cuda(), volatile=True)
        pred.append(model(x))
        act.append(y)
        
    pred = to_numpy(torch.cat(pred)).argmax(axis=1)
    act = to_numpy(torch.cat(act)).argmax(axis=1)
    
    return (pred == act).mean()

