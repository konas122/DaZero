import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import dazero
from dazero.models import ResNet50


url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dazero.utils.get_file(url)
img = Image.open(img_path)

x = ResNet50.img_preprocess(img)
x = x[np.newaxis]

model = ResNet50(pretrained=True)
with dazero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)
print(predict_id)
labels = dazero.datasets.ImageNet.labels()
print(labels[predict_id])

model.plot(x, to_file='resnet50.pdf')
