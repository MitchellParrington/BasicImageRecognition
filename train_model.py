import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import time
from logger import Logger

log = Logger(1)

def usage():
    log("\nUsage:", a=1)
    log(f"python {os.path.basename(__file__)} dataset-root-path -h -e -b [-vv | -v | -q]", a=1)
    log(f"\tdataset-root-path    : Path to root of a dataset folder. Should contain subfolders: 'Models', 'Training', 'Testing'", a=1)
    log(f"\t-h                   : Display help (this).", a=1)
    log(f"\t-e epochs            : Number of epochs model trains for. Default = 35", a=1)
    log(f"\t-b batch-size        : Size of bach used to train the model. Default = 32", a=1)
    log(f"\t-v                   : Verbose output (log to console). Default", a=1)
    log(f"\t-vv                  : Extra verbose output (logs everything).", a=1)
    log(f"\t-q                   : Quiet output (dont log to console).", a=1)
    log(f"", a=1)
    log("Abort", a=0)

try:
    epochs = 35 # default
    batch_size = 32 # default
    path = sys.argv[1]
    if not os.path.exists(path): raise FileNotFoundError
    if not os.path.exists(path+"\\Models"): raise FileNotFoundError
    if not os.path.exists(path+"\\Training"): raise FileNotFoundError
    if not os.path.exists(path+"\\Testing"): raise FileNotFoundError

    # Parse flags / parameters
    args = sys.argv
    if '-h' in args:
        log.v = 1
        leave(c)
    if '-vv' in args:
        log.v = 2
    elif '-v' in args:
        log.v = 1
    elif '-q' in args:
        log.v = 0
    if '-e' in args:
        par = args[args.index('-e') + 1]
        epochs = int(par)
    if '-b' in args:
        par = args[args.index('-b') + 1]
        batch_size = int(par)

    log(f"Initialising in 5 seconds. Press ctrl + c to cancel.", a=0)
    time.sleep(5)

except ValueError:
    log(f"Invalid conversion from str to int. Make sure intager values are being passed as numbers.", a=0)
    usage()
    exit()
except IndexError:
    log(f"Paramiter '{sys.argv[-1]}' expected an argument", a=0)
    usage()
    exit()
except FileNotFoundError:
    log(f"Invalid path. It may not exist or might be missing neccecary subfolders", a=0)
    usage()
    exit()
except KeyboardInterrupt:
    log(f"Abort", a=0)
    exit()

log(f"Train neural network using tensorflow and keras.\n", a=1)
log(f"Verbose output            : {log.v}", a=1)
log(f"Dataset path              : '{path}'", a=1)
log(f"Epochs                    : {epochs}", a=1)
log(f"Batch-Size                : {batch_size}", a=1)
log(a=1)

# Imports
log(f"Importing libraries...", a=1)
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils

# Load data
log("Loading data...", a=1)
X_train = np.load(f"{path}\\Training\\input.npy")
y_train = np.load(f"{path}\\Training\\output.npy")
X_test = np.load(f"{path}\\Testing\\input.npy")
y_test = np.load(f"{path}\\Testing\\output.npy")

train_shuffle = np.arange(len(X_train))
test_shuffle = np.arange(len(X_test))

np.random.shuffle(train_shuffle)
np.random.shuffle(test_shuffle)

X_train[:] = X_train[train_shuffle]#.astype('float32') / 255.0
y_train[:] = y_train[train_shuffle]
X_test[:] = X_test[test_shuffle]#.astype('float32') / 255.0
y_test[:] = y_test[test_shuffle]


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# for img, i in zip(X_train, y_train):
#     print(i)
#     plt.figure(figsize = (5,5))
#     plt.imshow(img)
#     plt.show()

class_num = y_test.shape[1]

log("Building network shape...", a=1)
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.2))

model.add(Dense(512, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, kernel_constraint=maxnorm(3)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(class_num))
model.add(Activation('softmax'))

# Compile model
log(f"Compiling model...", a=1)
optimizer = 'adam'
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.summary(print_fn=log)

# Train Model
log(f"Training model...\n", a=1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=log.v, use_multiprocessing=True)

# Model evaluation
log("Evaluating model...", a=1)
scores = model.evaluate(X_test, y_test, verbose=log.v)

log("Accuracy: %.2f%%" % (scores[1]*100), a=0)


plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.title('MAE for Chennai Reservoir Levels')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


try:
    log(f"Saving model in 5 seconds. Press ctrl + c to cancel.", a=0)
    time.sleep(5)
except KeyboardInterrupt:
    log("Abort", a=0)
    exit()

log(f"Saving model...", a=1)
ctime = str(time.time()).split(".")[0]
model.save(f"{path}\\Models\\model{ctime}_{int(scores[1]*100)}")
log(f"Model path: '{path}\\Models\\model{ctime}_{int(scores[1]*100)}'", a=0)
