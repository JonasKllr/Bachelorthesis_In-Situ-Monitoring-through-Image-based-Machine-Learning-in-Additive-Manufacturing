from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
import os
import pathlib

def load_images(file_path):
  image_camera = tf.io.read_file(file_path)
  image_camera = tf.image.decode_png(image_camera, channels=1)
  image_camera = tf.image.convert_image_dtype(image_camera, dtype= tf.float32)
  image_camera = tf.image.resize(image_camera, [140,310])

  # read in CAD path
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
  
  image_cad = tf.io.read_file(cad_path)
  image_cad = tf.image.decode_png(image_cad, channels=1)
  image_cad = tf.image.convert_image_dtype(image_cad, dtype= tf.float32)
  image_cad = tf.image.resize(image_cad, [140,310])

  return {'camera_image': image_camera, 'cad_image': image_cad}

# load camera images
data_dir = r'C:\PATH\TO\DATA'
data_dir = pathlib.Path(data_dir)
list_ds = tf.data.Dataset.list_files([str(data_dir/'*')], shuffle=False)

# load CAD images
cad_path_wuerfel = r'C:\PATH\TO\CAD_IMAGE'
cad_path_wuerfel_klein = r'C:\PATH\TO\CAD_IMAGE'
cad_path_rechteck_klein = r'C:\PATH\TO\CAD_IMAGE'

predict_ds = list_ds.map(load_images, deterministic=True)
predict_ds = predict_ds.batch(1)

model = tf.keras.models.load_model(r'C:\PATH\TO\TRAINED\MODEL')
prediciton = model.predict(predict_ds, verbose=1)

for i in range(1 ,len(prediciton)):
  print(i , prediciton[i][0], prediciton[i][1], prediciton[i][2], prediciton[i][3])

