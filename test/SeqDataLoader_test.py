import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import dazero
from dazero import SeqDataLoader


train_set = dazero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=3)

x, t = next(dataloader)
print(x)
print('-------------')
print(t)
