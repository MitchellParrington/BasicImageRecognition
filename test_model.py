import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sys import argv
from time import sleep
from logger import Logger
from keras.models import load_model
import numpy as np

log = Logger(1)

def usage():
    log(f"Usage:", a=1)
    log(f"\tpython {os.path.basename(__file__)} model-path data-path", a=1)
    log(f"Example:", a=1)
    log(f"\tpython {os.path.basename(__file__)} ./Datasets/x64/dataset/Models/model123456679_76 ./Datasets/x64/dataset/Training/input.npy", a=1)
    log(f"Optional tags '-vv', '-v', '-q' can be used as last argument to specify verbosity.", a=1)

# Check Args
try:
    if '-h' in argv:
        log.v = 1
        usage()
        exit()
    if '-vv' == argv[-1]:
        log.v = 2
    elif '-v' == argv[-1]:
        log.v = 1
    elif '-q' == argv[-1]:
        log.v = 0

    log("loading model...", a=1)
    model_path = argv[1]
    model = load_model(model_path)
    log("loading data...")
    data_path = argv[2]
    data = np.load(data_path)
    key_path = "/".join(data_path.split('/')[:-1]) + "/key.npy"
    key = np.load(key_path)

    if not os.path.exists(data_path): raise FileNotFoundError
    log(f"Initialising in 5 seconds. Press ctrl + c to cancel.", a=0)
    sleep(5)
except IndexError:
    log("Please provide all arguments.", a=0)
    usage()
    exit()
except ValueError:
    log("Intiger value passed improperly.", a=0)
    usage()
    exit()
except FileNotFoundError:
    log("Invalid destination path. Does it exist?", a=0)
    usage()
    exit()
except OSError:
    log("Invalid model path.", a=0)
    usage()
    exit()
except KeyboardInterrupt:
    log("Abort", a=0)
    exit()


# Import libries
log(f"Importing libries...", a=1)
import matplotlib.pyplot as plt

# Shuffle data
log(f"Shuffling data...", a=1)
shuffle = np.arange(len(data))
np.random.shuffle(shuffle)
data[:] = data[shuffle]

log([i[0] for i in key[::-1]], a=0)

try:
    for img in data:
        s = model.predict(np.array([img,]))
        log((s*100).astype(int), "     ", end='\r', a=0)
        plt.figure(figsize = (5,5))
        plt.imshow(img)
        plt.show()
except KeyboardInterrupt:
    log(f"\nAbort.", a=0)

log(a=0)
