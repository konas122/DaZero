import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from header import *

import numpy as np
import matplotlib.pyplot as plt

import dazero
from dazero import Model
from dazero import SeqDataLoader
import dazero.functions as F
import dazero.layers as L


max_epoch = 30
batch_size = 30
hidden_size = 100
bptt_length = 30

train_set = dazero.datasets.SinCurve(train=True)
dataloader = SeqDataLoader(train_set, batch_size=batch_size)
seqlen = len(train_set)


class BetterRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.LSTM(1, hidden_size)
        self.fc = L.Linear(hidden_size, out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def __call__(self, x):
        y = self.rnn(x)
        y = self.fc(y)
        return y


model = BetterRNN(hidden_size, 1)
optimizer = dazero.optimizers.Adam(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in dataloader:
        y = model(x)
        loss += F.mse_loss(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.zero_grad()
            loss.backward()
            loss.detach()
            optimizer.step()
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

# Plot
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dazero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
