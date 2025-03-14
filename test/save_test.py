import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from urllib.error import HTTPError

# import dazero
import dazero.layers as L
import dazero.functions as F
# from dazero import DataLoader
from dazero import optimizers, Model


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


max_epoch = 1
batch_size = 100

# try:
#     train_set = dazero.datasets.MNIST(train=True)
#     train_loader = DataLoader(train_set, batch_size)
# except HTTPError as e:
#     print("Failed to download the datasets.", file=sys.stderr)
#     print(e, file=sys.stderr)
#     sys.exit(0)

model = MLP(784, (1000, 10))
optimizer = optimizers.SGD(model)

if os.path.exists('my_mlp.npz'):
    model.load_weights('my_mlp.npz')

# for epoch in range(max_epoch):
#     sum_loss = 0

#     for x, t in train_loader:
#         y = model(x)
#         loss = F.softmax_cross_entropy(y, t)
#         model.zero_grad()
#         loss.backward()
#         optimizer.step()
#         sum_loss += float(loss.data) * len(t)

#     print('epoch: {}, loss: {:.4f}'.format(
#         epoch + 1, sum_loss / len(train_set)))

model.save_weights('my_mlp.npz')
