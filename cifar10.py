
import numpy as np
import os

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import (ModelCheckpoint,
                             LearningRateScheduler,
                             TensorBoard)

from resnet import ResNet18, ResNet34
from adabound import AdaBound

# Training parameters
batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True
num_classes = 10

# adabound parameters
adabound_final_lr = 0.1
adabound_gamma = 1e-3
weight_decay = 5e-4
amsbound = False

"""
Following adapted from https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model version
n = 34  # [can be 18 or 34]

assert n in (18, 34), "N must be 18 or 34"

depth = n
version = 1

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.001
    epoch += 1
    # if epoch >= 90:
    #     lr *= 5e-2
    # elif epoch >= 60:
    #     lr *= 1e-1
    # elif epoch >= 30:
    #     lr *= 5e-1
    if epoch >= 150:
        lr *= 0.1
    print('Learning rate: ', lr)
    return lr

if n == 18:
    model = ResNet18(input_shape=input_shape, depth=depth)
else:
    model = ResNet34(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=AdaBound(lr=lr_schedule(0),
                                 final_lr=adabound_final_lr,
                                 gamma=adabound_gamma,
                                 weight_decay=weight_decay,
                                 amsbound=amsbound),
              metrics=['accuracy'])
model.summary()
print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'weights')
model_name = 'cifar10_%s_model.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)


log_path = 'logs/%s' % (filepath[:-3])
tensorboard = TensorBoard(log_path, update_freq='batch')

callbacks = [checkpoint, lr_scheduler, tensorboard]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.,
        # randomly shift images vertically
        height_shift_range=0.,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1,
                        steps_per_epoch=x_train.shape[0] // batch_size + 1,
                        callbacks=callbacks)

model.load_weights(filepath)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
