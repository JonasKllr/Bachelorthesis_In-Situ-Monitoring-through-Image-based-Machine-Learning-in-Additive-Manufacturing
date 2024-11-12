from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from pathlib import Path
from datetime import datetime
import re
import sys


# data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Wuerfel_layershift_vorne_12_red_bearbeitet_number_geloescht'
# data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Wuerfel_Spaghetti_image_number_geloescht'
# data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Wuerfel_layershift_vorne_10_bearbeitet_number_geloescht'
# data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Wuerfel_clogged_nozzle_black_number_geloescht'
data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Wuerfel_clogged_nozzle_silver_number_geloescht'
# data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Wuerfel_klein_number_geloescht'
# data_dir = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Bearbeitet_number\Evaluation\Bilder_geloescht\Rechteck_klein_number_geloescht'


data_dir = pathlib.Path(data_dir)

# Dataset aus nur einem Bild erstellen
list_ds = tf.data.Dataset.list_files([str(data_dir/'*')], shuffle=False)

for f in list_ds.take(20):
  print(f.numpy())

# Pfad zu CAD Bild    ACHTUNG CAD BILDER IN DATENSATZ_UNSEEN FALSCH
cad_path_wuerfel = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Blender\Wuerfel_render.png'
cad_path_wuerfel_klein = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Blender\Wuerfel_klein_render.png'
cad_path_rechteck_klein = r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\3D_Druck\Bilder\Datensatz_raw\Blender\Rechteck_klein_render.png'

# Bilder laden
def load_images(file_path):

  #Kamerabild laden 
  image_camera = tf.io.read_file(file_path)
  image_camera = tf.image.decode_png(image_camera, channels=1)
  image_camera = tf.image.convert_image_dtype(image_camera, dtype= tf.float32)
  image_camera = tf.image.resize(image_camera, [140,310])

  # Pfad zu CAD Bild einlesen
  def wuerfel(): return tf.constant(cad_path_wuerfel)
  def wuerfel_klein(): return tf.constant(cad_path_wuerfel_klein)
  def rechteck_klein(): return tf.constant(cad_path_rechteck_klein)

  cad_path = tf.case([
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_clogged_nozzle_black_number_geloescht'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_clogged_nozzle_silver_number_geloescht'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_Spaghetti_image_number_geloescht'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_10_bearbeitet_number_geloescht'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_layershift_vorne_12_red_bearbeitet_number_geloescht'), wuerfel),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Wuerfel_klein_number_geloescht'), wuerfel_klein),
                      (tf.strings.split(file_path, os.path.sep)[-2] == tf.constant('Rechteck_klein_number_geloescht'), rechteck_klein),
                      ])
  
  #CAD-Bild laden
  image_cad = tf.io.read_file(cad_path)
  image_cad = tf.image.decode_png(image_cad, channels=1)
  image_cad = tf.image.convert_image_dtype(image_cad, dtype= tf.float32)
  image_cad = tf.image.resize(image_cad, [140,310])

  # tf.print(file_path)

  return {'camera_image': image_camera, 'cad_image': image_cad}


# Dataset bearbeiten
predict_ds = list_ds.map(load_images, deterministic=True)

# Dataset in Batchsize 
predict_ds = predict_ds.batch(1)


# gespeichertes Netz laden
model = tf.keras.models.load_model(r'C:\Users\Jonas\Documents\Studium\Bachelor\Bachelorarbeit\Python\Klassifikation\Training_output_Klassifikation\Training_output_Klassifikation_pfad\Modell_speicher')
# model.summary()


# Vorhersage für Dataset treffen
pred = model.predict(predict_ds, verbose=1)


lenth = len(pred)

for i in range(1 ,lenth):
  print(i , pred[i][0], pred[i][1], pred[i][2], pred[i][3])

