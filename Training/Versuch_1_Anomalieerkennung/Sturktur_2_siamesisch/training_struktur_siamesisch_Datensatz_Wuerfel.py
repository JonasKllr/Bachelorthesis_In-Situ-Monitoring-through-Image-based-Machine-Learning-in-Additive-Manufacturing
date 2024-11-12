#https://www.tensorflow.org/guide/data#preprocessing_data

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from pathlib import Path
from datetime import datetime


AUTOTUNE = tf.data.experimental.AUTOTUNE

# Ordner mit Inhalt Struktur der Daten
data_dir = r'/fzi/ids/qy134/no_backup/Datensatz_Wuerfel/train'
data_dir = pathlib.Path(data_dir)     #liest den Pfad ein

# Labels in Array schreiben
label_dir = data_dir / 'labels'
label_names = np.array([item.name for item in label_dir.glob('*')])

# Dataset aus allen Dateipfaden anlegen 
list_ds = tf.data.Dataset.list_files([str(data_dir/'labels/*/*/*')])    

# alle Pfade zu CAD-Bildern
cad_path_wuerfel = r'/fzi/ids/qy134/no_backup/Datensatz_Wuerfel/train/cad/Wuerfel.png'


# Label abh. von Kamerabild einlesen. (gut=0 / schlecht=1)
def get_label(file_path):

  parts = tf.strings.split(file_path, os.path.sep)

  def gut(): return tf.constant(0)
  def schlecht(): return tf.constant(1)

  label = tf.case([(parts[-3] == tf.constant('gut'), gut),
                   (parts[-3] == tf.constant('schlecht'), schlecht)]) 

  return label                    


# set (Kamerabild, CAD_Bild, Label) erstellen 
def load_images(file_path):
  
  # Label einlesen
  label = get_label(file_path)
  
  # Kamerabild laden 
  image_camera = tf.io.read_file(file_path)
  image_camera = tf.image.decode_png(image_camera, channels=1)
  image_camera = tf.image.convert_image_dtype(image_camera, dtype= tf.float32)
  image_camera = tf.image.resize(image_camera, [140,310])

  # Passenden CAD-Pfad zu image_camera aussuchen
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
  
  # CAD-Bild laden
  image_cad = tf.io.read_file(cad_path)
  image_cad = tf.image.decode_png(image_cad, channels=1)
  image_cad = tf.image.convert_image_dtype(image_cad, dtype= tf.float32)
  image_cad = tf.image.resize(image_cad, [140,310])
  
  return {'camera_image': image_camera, 'cad_image': image_cad}, {'voll_connected': label}     


def augmentation(file_path):
  
  # Label einlesen
  label = get_label(file_path)
  
  # Kamerabild laden 
  image_camera = tf.io.read_file(file_path)
  image_camera = tf.image.decode_png(image_camera, channels=1)
  image_camera = tf.image.convert_image_dtype(image_camera, dtype= tf.float32)
  image_camera = tf.image.resize(image_camera, [140,310])
  image_camera = tf.image.random_brightness(image_camera, max_delta=0.2)

  # Passenden CAD-Pfad zu image_camera aussuchen
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
  
  # CAD-Bild laden
  image_cad = tf.io.read_file(cad_path)
  image_cad = tf.image.decode_png(image_cad, channels=1)
  image_cad = tf.image.convert_image_dtype(image_cad, dtype= tf.float32)
  image_cad = tf.image.resize(image_cad, [140,310])
  
  return {'camera_image': image_camera, 'cad_image': image_cad}, {'voll_connected': label}



labeled_ds = list_ds.map(load_images, num_parallel_calls=AUTOTUNE)
augmentet_ds = list_ds.map(augmentation, num_parallel_calls=AUTOTUNE)
complete_ds = labeled_ds.concatenate(augmentet_ds)

# Dataset shuffle
complete_ds = complete_ds.shuffle(tf.data.experimental.cardinality(complete_ds).numpy(), reshuffle_each_iteration=True)



# Dataset in Trainigs-, Validierungs-, Testdaten teilen
DATASET_SIZE = tf.data.experimental.cardinality(complete_ds).numpy()
train_size = int(0.8 * DATASET_SIZE)
val_size = int(0.2 * DATASET_SIZE)

train_ds = complete_ds.take(train_size)
val_ds = complete_ds.skip(train_size)

# Datasets in Batches aufteilel
BATCH_SIZE = 2
train_ds = train_ds.batch(BATCH_SIZE)
val_ds = val_ds.batch (1)
  


############################## Netz ##############################
img_heigth = 140
img_width = 310

# Inputlayers
cam_input = tf.keras.Input(shape=(img_heigth, img_width, 1), name='camera_image')
cad_input = tf.keras.Input(shape=(img_heigth, img_width, 1), name='cad_image')

# Netz Kamera
cam_features = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_input)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cam_features)
cam_features = tf.keras.layers.Dropout(0.5)(cam_features)

cam_features = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cam_features)
cam_features = tf.keras.layers.Dropout(0.5)(cam_features)

cam_features = tf.keras.layers.Conv2D(filters=40, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.Conv2D(filters=40, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cam_features)
cam_features = tf.keras.layers.Dropout(0.5)(cam_features)

cam_features = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)
cam_features = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cam_features)
cam_features = tf.keras.layers.BatchNormalization()(cam_features)



# Netz CAD
cad_features = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_input)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.Conv2D(filters=10, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cad_features)
cad_features = tf.keras.layers.Dropout(0.5)(cad_features)

cad_features = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.Conv2D(filters=20, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cad_features)
cad_features = tf.keras.layers.Dropout(0.5)(cad_features)

cad_features = tf.keras.layers.Conv2D(filters=40, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.Conv2D(filters=40, kernel_size=3, strides=(1, 1), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(cad_features)
cad_features = tf.keras.layers.Dropout(0.5)(cad_features)

cad_features = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)
cad_features = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(2, 2), padding='same', activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(cad_features)
cad_features = tf.keras.layers.BatchNormalization()(cad_features)



# Pfade zusammenfeuhren 
x = tf.keras.layers.concatenate([cam_features, cad_features])

x = tf.keras.layers.Conv2D(filters=160, kernel_size=3, strides=(2, 2), padding='same', activation='elu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)

x = tf.keras.layers.Conv2D(filters=80, kernel_size=3, strides=(1, 1), padding='same', activation='elu')(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Flatten()(x)

x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(160, activation='elu')(x)

fully_connected = tf.keras.layers.Dense(1, name='voll_connected')(x)       

# Modell bauen
model = tf.keras.Model(inputs=[cam_input, cad_input], outputs=fully_connected)

model.compile(
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy()]
              )


# Model Plot
# tf.keras.utils.plot_model(model, to_file='model_aktuell.png', show_shapes=True)
model.summary()

# Tensorboard callback
log_dir = r'/fzi/ids/qy134/no_backup/Training_output/Tensorboard/Date' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=2)

#Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Training + Validiierung 
model.fit(train_ds, epochs = 30, validation_data= val_ds, callbacks= [tensorboard_callback, early_stopping])


# komplettes Modell speichern und aus diesem Skript loeschen
model.save(r'/fzi/ids/qy134/no_backup/Training_output/Modell_speicher', overwrite= True)
del model
