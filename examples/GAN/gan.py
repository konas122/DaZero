import time
import numpy as np
import matplotlib.pyplot as plt

import dazero
import dazero.layers as L
import dazero.functions as F
from dazero import DataLoader
from dazero.models import Sequential
from dazero.optimizers import Adam


max_epoch = 5
batch_size = 128
hidden_size = 62
use_gpu = dazero.cuda.gpu_enable

fc_channel, fc_height, fc_width = 128, 7, 7


def init_weight(dis, gen, hidden_size):
    # Input dummy data to initialize weights
    batch_size = 1
    z = np.random.rand(batch_size, hidden_size)
    fake_images = gen(z)
    dis(fake_images)

    for l in dis.layers + gen.layers:
        classname = l.__class__.__name__
        if classname.lower() in ('conv2d', 'linear', 'deconv2d'):
            l.W.data = 0.02 * np.random.randn(*l.W.data.shape)


gen = Sequential(
    L.Linear(62, 1024),
    L.BatchNorm2d(),
    F.relu,
    L.Linear(1024, fc_channel * fc_height * fc_width),
    L.BatchNorm2d(),
    F.relu,
    lambda x: F.reshape(x, (-1, fc_channel, fc_height, fc_width)),
    L.Deconv2d(fc_channel, fc_channel // 2, kernel_size=4, stride=2, pad=1),
    L.BatchNorm2d(),
    F.relu,
    L.Deconv2d(fc_channel // 2, 1, kernel_size=4, stride=2, pad=1),
    F.sigmoid
)

dis = Sequential(
    L.Conv2d(1, 64, kernel_size=4, stride=2, pad=1),
    F.leaky_relu,
    L.Conv2d(64, 128, kernel_size=4, stride=2, pad=1),
    L.BatchNorm2d(),
    F.leaky_relu,
    F.flatten,
    L.Linear(128 * 7 * 7, 1024),
    L.BatchNorm2d(),
    F.leaky_relu,
    L.Linear(1024, 1),
    F.sigmoid
)

init_weight(dis, gen, hidden_size)

opt_g = Adam(gen, alpha=0.0002, beta1=0.5)
opt_d = Adam(dis, alpha=0.0002, beta1=0.5)

transform = lambda x: (x / 255.0).astype(np.float32)
train_set = dazero.datasets.MNIST(train=True, transform=transform)
train_loader = DataLoader(train_set, batch_size)

if use_gpu:
    gen.to_gpu()
    dis.to_gpu()
    train_loader.to_gpu()
    xp = dazero.cuda.cupy
    print("Using GPU")
else:
    xp = np
    print("Using CPU")

label_real = xp.ones(batch_size).astype(xp.int16)
label_fake = xp.zeros(batch_size).astype(xp.int16)
test_z = xp.random.randn(25, hidden_size).astype(xp.float32)


def generate_image(idx):
    with dazero.test_mode():
        fake_images = gen(test_z)

    img = dazero.cuda.as_numpy(fake_images.data)
    plt.figure()
    for i in range(0, 25):
        ax = plt.subplot(5, 5, i+1)
        ax.axis('off')
        plt.imshow(img[i][0], 'gray')
    plt.show()
    # plt.savefig('gan_{}.png'.format(idx))

start = time.time()
for epoch in range(max_epoch):
    avg_loss_d = 0
    avg_loss_g = 0
    cnt = 0

    for x, t in train_loader:
        cnt += 1
        if len(t) != batch_size:
            continue

        # (1) Update discriminator
        z = xp.random.randn(batch_size, hidden_size).astype(np.float32)
        fake = gen(z)
        y_real = dis(x)
        y_fake = dis(fake.data)
        loss_d = F.binary_cross_entropy(y_real, label_real) + \
                 F.binary_cross_entropy(y_fake, label_fake)
        gen.zero_grad()
        dis.zero_grad()
        loss_d.backward()
        opt_d.step()

        # (2) Update generator
        y_fake = dis(fake)
        loss_g = F.binary_cross_entropy(y_fake, label_real)
        gen.zero_grad()
        dis.zero_grad()
        loss_g.backward()
        opt_g.step()

        # Print loss & visualize generator
        avg_loss_g += loss_g.data
        avg_loss_d += loss_d.data
        interval = 100 if use_gpu else 5
        if cnt % interval == 0:
            epoch_detail = epoch + cnt / train_loader.max_iter
            print('epoch: {:.2f},\tloss_g: {:.4f},\tloss_d: {:.4f}'.format(
                epoch_detail, float(avg_loss_g/cnt), float(avg_loss_d/cnt)))
        
        interval = 100 if use_gpu else 50
        if cnt % interval == 0:
            generate_image(cnt)

end = time.time()
print("Using GPU: ", dazero.cuda.gpu_enable)
print("Training for ", end - start, 's', sep="")
