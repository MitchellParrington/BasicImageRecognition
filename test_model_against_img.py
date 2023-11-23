import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from convert_jpg_to_npy import convert_single
from keras.models import load_model
import matplotlib.pyplot as plt
from logger import Logger
log = Logger(1)

model = load_model(sys.argv[1])
img = convert_single(sys.argv[2], r=False)
dimg = convert_single(sys.argv[2], r=False, resolution=1028)

s = model.predict(np.array([img,]))
log((s*100).astype(int), a=0)
plt.figure(figsize = (5,5))
plt.imshow(dimg)
plt.show()
