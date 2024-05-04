import math
import numpy as np
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False

from dazero import Dataset, cuda


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False, drop_last=False):
        if not isinstance(dataset, Dataset):
            raise TypeError("It must be of type dazero.Dataset")
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        
        if drop_last:
            self.max_iter = self.data_size // batch_size
        else:
            self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu

        self._reset()

    def _reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self._reset()
            raise StopIteration()

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size: (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        if cuda.gpu_enable == False:
            self.gpu = False
            return
        self.gpu = True


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)
        self.jump = self.data_size // self.batch_size

    def __next__(self):
        if self.iteration >= self.max_iter:
            self._reset()
            raise StopIteration

        batch_index = [(i * self.jump + self.iteration) % self.data_size for i in range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
