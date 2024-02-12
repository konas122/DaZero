import math
pil_available = True
try:
    from PIL import Image
except:
    pil_available = False
import numpy as np

from dazero import Dataset

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        if not isinstance(dataset, Dataset):
            raise TypeError("It must be of type dazero.Dataset")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

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
        batch_index = self.index[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]

        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()
        
