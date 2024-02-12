import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from header import *

import dazero
from dazero import Model
import dazero.layers as L
import dazero.functions as F
from dazero import DataLoader


max_epoch = 2
batch_size = 100
hidden_size = 1000


class MLP(Model):
    def __init__(self, in_dims, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []
        self.in_dims = in_dims

        for i, out_size in enumerate(fc_output_sizes):
            if i == 0:
                layer = L.Linear(in_dims, out_size)
            else:
                layer = L.Linear(fc_output_sizes[i - 1], out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


train_set = dazero.datasets.MNIST(train=True)
test_set = dazero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP(784, (hidden_size, hidden_size, 10), activation=F.relu)
optimizer = dazero.optimizers.Adam(model)
optimizer.add_hook(dazero.optimizers.WeightDecay(1e-4))  # Weight decay


for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('\ttrain loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dazero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('\ttest  loss: {:.4f}, accuracy: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))
