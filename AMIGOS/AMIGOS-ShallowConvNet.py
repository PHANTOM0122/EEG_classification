import numpy as np
from scipy import io
import os
import random
# EEGNet-specific imports
from EEGModels import DeepConvNet, ShallowConvNet
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D

########################################################################################################################
# data load
# Train data load
data_path = '/home/hj/PycharmProjects/EEG/dataset/Amigos/data_preprocessed_python_1s_norm/'
train_subject = [1, 2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32, 34, 36, 37, 38,
                 39]  # 27 train subjects
test_subject = [4, 8, 25, 26, 35, 40]  # 6 test subject
num_trial = 16  # for short vidoes only

X_train = []
y_train = []
X_test = []
y_test = []

for i in train_subject:
    # original eeg dataset(17): 1~14: eeg / 15, 16: EOG / 17:GSR
    # Only EEG signals are extracted and transpose
    for j in range(1, num_trial + 1):
        k = 0
        while True:
            # Read file
            eeg_file = data_path + 'S%02dT%02d_%04d.npy' % (i, j, k)
            label_file = data_path + 'S%02dT%02d_%04d_arousal.txt' % (i, j, k)
            if not os.path.exists(eeg_file):
                break
            # Add data
            data = np.load(eeg_file)
            self_assment = np.loadtxt(label_file)

            # arousal
            if self_assment[0] >= 5:
                self_assment[0] = 1
            else:
                self_assment[0] = 0
            '''
            # valence
            if self_assment[1] > 5:
                self_assment[1] = 1
            else:
                self_assment[1] = 0
            '
            # If label isn't one-hot encoded pass
            sum = 0
            for elem in self_assment[0]:
                sum += elem
            if sum != 1:
                k += 1
                continue
            else:
                y_train.append(self_assment[0])
                X_train.append(data)
                k += 1
            '''
            y_train.append(self_assment[0])
            X_train.append(data)
            k += 1
# Test data load
for i in test_subject:
    # original eeg dataset(17): 1~14: eeg / 15, 16: EOG / 17:GSR
    # Only EEG signals are extracted and transpose
    for j in range(1, num_trial + 1):
        k = 0
        while True:
            # Read file
            eeg_file = data_path + 'S%02dT%02d_%04d.npy' % (i, j, k)
            label_file = data_path + 'S%02dT%02d_%04d_arousal.txt' % (i, j, k)
            if not os.path.exists(eeg_file):
                break
            # Add data
            data = np.load(eeg_file)
            self_assment = np.loadtxt(label_file)

            # arousal
            if self_assment[0] >= 5:
                self_assment[0] = 1
            else:
                self_assment[0] = 0
            '''
            # valence
            if self_assment[1] > 5:
                self_assment[1] = 1
            else:
                self_assment[1] = 0

            # If label isn't one-hot encoded pass
            sum = 0
            for elem in self_assment[5:7]:
                sum += elem
            if sum != 1:
                k += 1
                continue
            else:
                y_test.append(self_assment[0])
                X_test.append(data)
                k += 1
            '''
            y_test.append(self_assment[0])
            X_test.append(data)
            k += 1
########################################################################################################################
# X Data format : channel X samples X kernel(==1)
channel = 14
sample = 128
num_classes = 2
# random.shuffle(EEG_data)

X_train = np.array(X_train).astype('float64')
X_train = X_train.reshape(-1, channel, sample, 1)
X_test = np.array(X_test).astype('float64')
X_test = X_test.reshape(-1, channel, sample, 1)

y_train = np.array(y_train)
y_train = y_train.reshape(-1, 1)
y_test = np.array(y_test)
y_test = y_test.reshape(-1, 1)

print('X_train.shape', X_train.shape)
print('X_test.shape', X_test.shape)
print('y_train.shape', y_train.shape)
print('y_test.shape', y_test.shape)
########################################################################################################################
# DeepConvNet
model = ShallowConvNet(nb_classes=num_classes, Chans=channel, Samples=sample, dropoutRate=0.5)

# compile the model and set the optimizers
opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-8)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Class weight
# class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

# Save best model
Model_save_path = '/home/hj/PycharmProjects/EEG/Amigos'
if not os.path.exists(Model_save_path):
    os.mkdir(Model_save_path)
Model_path = Model_save_path + 'ShallowConvNet(AMIGOS).hdf5'

# Callback
patient = 5
callbacks_list = [
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.1,
        patience=patient,
        min_lr=0.00001,
        verbose=1,
        mode='max'
    ),
    ModelCheckpoint(
        filepath=Model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1,
        mode='max'
    )]

# Train model
model.fit(X_train, y_train, batch_size=16, epochs=100,
          verbose=1, validation_split=0.2,
          callbacks=callbacks_list)
print("model is fitted!")

# Load best model
model.load_weights(Model_path)
print("Best model is loaded!\n")
########################################################################################################################
# Predict model
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=16, verbose=1)
print('test acc:', test_acc, 'test_loss:', test_loss)


