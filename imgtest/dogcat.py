import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
pathZip = tf.keras.utils.get_file(
    'cats_and_dogs.zip', origin=URL, extract=True)
PATH = os.path.join(os.path.dirname(pathZip), 'cats_and_dogs_filtered')

tr_dir = os.path.join(PATH, 'train')
va_dir = os.path.join(PATH, 'validation')

tr_dogs = os.path.join(tr_dir, 'dogs')
tr_cats = os.path.join(tr_dir, 'cats')

va_dogs = os.path.join(va_dir, 'dogs')
va_cats = os.path.join(va_dir, 'cats')

num_tr_dogs = len(os.listdir(tr_dogs))
num_tr_cats = len(os.listdir(tr_cats))

num_va_dogs = len(os.listdir(va_dogs))
num_va_cats = len(os.listdir(va_cats))

to_tr = num_tr_dogs + num_tr_cats
to_va = num_va_dogs + num_va_cats

print("dogs' training images : ", num_tr_dogs)
print("cats' training images : ", num_tr_cats)

print("dogs' valid images : ", num_va_dogs)
print("cats' valid images : ", num_va_cats)

print("total training images : ", to_tr)
print("total valid images : ", to_va)

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

tr_img_gen = ImageDataGenerator(rescale=1./255)
va_img_gen = ImageDataGenerator(rescale=1./255)

tr_data_gen = tr_img_gen.flow_from_directory(batch_size=batch_size,
                                             directory=tr_dir,
                                             shuffle=True,
                                             target_size=(
                                                 IMG_HEIGHT, IMG_WIDTH),
                                             class_mode='binary')
va_data_gen = va_img_gen.flow_from_directory(batch_size=batch_size,
                                             directory=va_dir,
                                             target_size=(
                                                 IMG_HEIGHT, IMG_WIDTH),
                                             class_mode='binary')
tr_sam_img, _ = next(tr_data_gen)


def pl_img(img_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for i, ax in zip(img_arr, axes):
        ax.imshow(i)
        ax.axis('off')
    # plt.tight_layout()
    # plt.show()


pl_img(tr_sam_img[:5])

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu',
           input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(
    from_logits=True), metrics=['accuracy'])

result = model.fit_generator(
    tr_data_gen,
    steps_per_epoch=to_tr//batch_size,
    epochs=epochs,
    validation_data=va_data_gen,
    validation_steps=to_va//batch_size
)

acc = result.history['accuracy']
va_acc = result.history['val_accuracy']
loss = result.history['loss']
va_loss = result.history['val_loss']

ep_ran = range(epochs)

plt.figure(figsize=(8, 8))
