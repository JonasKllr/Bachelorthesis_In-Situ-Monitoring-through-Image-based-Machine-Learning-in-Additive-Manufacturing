#https://www.tensorflow.org/guide/data#preprocessing_data

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
import numpy as np
import os
import pathlib
from datetime import datetime


def load_images(file_path):
  label = get_label(file_path)
  
  image_camera = tf.io.read_file(file_path)
  image_camera = tf.image.decode_png(image_camera, channels=1)
  image_camera = tf.image.convert_image_dtype(image_camera, dtype= tf.float32)
  image_camera = tf.image.resize(image_camera, [140,310])

  # get path to CAD image
  def wuerfel(): return tf.constant(cad_path_wuerfel)

  cad_path = tf.case([
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_black'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_clogged_nozzle'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_hinten'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_hinten_13'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_links'), wuerfel),                   
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_rechts'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_rechts_15'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_10'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Kugel_Spaghetti'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Motek_geloest_1'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_Spaghetti'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_hinten_6'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_links_15'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_5'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_16'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_links_10'), wuerfel),
                      ])
  
  image_cad = tf.io.read_file(cad_path)
  image_cad = tf.image.decode_png(image_cad, channels=1)
  image_cad = tf.image.convert_image_dtype(image_cad, dtype= tf.float32)
  image_cad = tf.image.resize(image_cad, [140,310])
  
  return ({'camera_image': image_camera, 'cad_image': image_cad}, {'dense': label})


def augmentation(file_path):
  label = get_label(file_path)
  
  image_camera = tf.io.read_file(file_path)
  image_camera = tf.image.decode_png(image_camera, channels=1)
  image_camera = tf.image.convert_image_dtype(image_camera, dtype= tf.float32)
  image_camera = tf.image.resize(image_camera, [140,310])
  
  # image sementation
  image_camera = tf.image.random_brightness(image_camera, max_delta=0.1)

  # get path to CAD image
  def wuerfel(): return tf.constant(cad_path_wuerfel)

  cad_path = tf.case([
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_black'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_clogged_nozzle'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_hinten'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_hinten_13'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_links'), wuerfel),                   
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_rechts'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_rechts_15'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_10'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Kugel_Spaghetti'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Motek_geloest_1'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_Spaghetti'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_hinten_6'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_links_15'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_5'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_16'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_links_10'), wuerfel),
                      ])
  
  image_cad = tf.io.read_file(cad_path)
  image_cad = tf.image.decode_png(image_cad, channels=1)
  image_cad = tf.image.convert_image_dtype(image_cad, dtype= tf.float32)
  image_cad = tf.image.resize(image_cad, [140,310])
  
  return ({'camera_image': image_camera, 'cad_image': image_cad}, {'dense': label})  


def get_label(file_path):
  def gut(): return tf.constant(0)
  def schlecht(): return tf.constant(1)

  parts = tf.strings.split(file_path, os.path.sep)
  label = tf.case([(parts[-3] == tf.constant('gut'), gut),
                   (parts[-3] == tf.constant('schlecht'), schlecht)]) 
  return label



# paths to CAD images
cad_path_wuerfel = r'/PATH/TO/CAD/IMAGE'

data_dir = r'/PATH/TO/DATA'
data_dir = pathlib.Path(data_dir)

label_dir = data_dir / 'labels'
label_names = np.array([item.name for item in label_dir.glob('*')])

list_ds = tf.data.Dataset.list_files([str(data_dir/'labels/*/*/*')])    

AUTOTUNE = tf.data.experimental.AUTOTUNE
labeled_ds = list_ds.map(load_images, num_parallel_calls=AUTOTUNE)
augmentet_ds = list_ds.map(augmentation, num_parallel_calls=AUTOTUNE)
complete_ds = labeled_ds.concatenate(augmentet_ds)
complete_ds = complete_ds.shuffle(tf.data.experimental.cardinality(complete_ds).numpy(), reshuffle_each_iteration=True)

DATASET_SIZE = tf.data.experimental.cardinality(complete_ds).numpy()
train_size = int(0.8 * DATASET_SIZE)
train_ds = complete_ds.take(train_size)
val_ds = complete_ds.skip(train_size)

BATCH_SIZE = 2
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch(1)

############################## CNN ##############################
img_heigth = 140
img_width = 310

# Inputlayers
cam_input = tf.keras.Input(shape=(img_heigth, img_width, 1), name='camera_image')
cad_input = tf.keras.Input(shape=(img_heigth, img_width, 1), name='cad_image')

# concatenate
x = tf.keras.layers.concatenate([cam_input, cad_input])

x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.Conv2D(filters=40, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=40, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Dropout(0.25)(x)

x = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.Dropout(0.25)(x)


x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(240, activation='elu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
fully_connected = tf.keras.layers.Dense(1, name='dense')(x)

# Modell bauen
model = tf.keras.Model(inputs=[cam_input, cad_input], outputs=fully_connected)

model.compile(
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
              loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()]
              )
model.summary()

# callbacks
log_dir = r'/PATH/Date' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

model.fit(train_ds, epochs = 30, validation_data= val_ds, callbacks= [tensorboard_callback, early_stopping])
model.save(r'/PATH/TO/SAVE', overwrite= True)
del model
