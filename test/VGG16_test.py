import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import dazero
from dazero.models import VGG16


url = 'https://github.com/oreilly-japan/deep-learning-from-scratch-3/raw/images/zebra.jpg'
img_path = dazero.utils.get_file(url)
img = Image.open(img_path)

x = VGG16.img_preprocess(img)
x = x[np.newaxis]

model = VGG16(pretrained=True)
with dazero.test_mode():
    y = model(x)
predict_id = np.argmax(y.data)
print(predict_id)
model.plot(x, to_file='vgg16.pdf')
