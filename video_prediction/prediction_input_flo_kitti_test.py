# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for building the input for the prediction model."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import flags
from tensorflow.python.platform import gfile

FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 1224
ORIGINAL_HEIGHT = 370
RESIZE_WIDTH = 1216
RESIZE_HEIGHT = 384
COLOR_CHAN = 3

def augment_image_pair(left_image, right_image):
  # randomly shift gamma
  random_gamma = tf.random_uniform([], 0.8, 1.2)
  left_image_aug  = left_image  ** random_gamma
  right_image_aug = right_image ** random_gamma

  # randomly shift brightness
  random_brightness = tf.random_uniform([], 0.5, 2.0)
  left_image_aug  =  left_image_aug * random_brightness
  right_image_aug = right_image_aug * random_brightness
 
  # randomly shift color
  random_colors = tf.random_uniform([3], 0.8, 1.2)
  white = tf.ones([tf.shape(left_image)[0], tf.shape(left_image)[1]])
  color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
  left_image_aug  *= color_image
  right_image_aug *= color_image
  
  left_image_aug += tf.random_normal(shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN], stddev=0.1)
  right_image_aug += tf.random_normal(shape=[ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN], stddev=0.1)
  
  # saturate
  left_image_aug  = tf.clip_by_value(left_image_aug,  0, 1)
  right_image_aug = tf.clip_by_value(right_image_aug, 0, 1)

  return left_image_aug, right_image_aug

def build_tfrecord_input(training=True, num_epochs=None):
  """Create input tfrecord tensors.

  Args:
    training: training or validation data.
  Returns:
    list of tensors corresponding to images, actions, and states. The images
    tensor is 5D, batch x time x height x width x channels. The state and
    action tensors are 3D, batch x time x dimension.
  Raises:
    RuntimeError: if no files found.
  """
  train_2012_filenames = gfile.Glob(os.path.join("/home/wangyang59/Data/ILSVRC2016_tf_kitti_2012_train_hist", '*'))
  train_2015_filenames = gfile.Glob(os.path.join("/home/wangyang59/Data/ILSVRC2016_tf_kitti_2015_train_hist", '*'))
  val_2015_filenames = gfile.Glob(os.path.join("/home/wangyang59/Data/ILSVRC2016_tf_kitti_2015_val_hist_fullsize", '*'))
  
  if training:
    filenames = val_2015_filenames
  else:
    filenames = val_2015_filenames
    #filenames = filenames[:index]
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = {"image1_raw": tf.FixedLenFeature([1], tf.string),
              "image2_raw": tf.FixedLenFeature([1], tf.string),
              "scene": tf.FixedLenFeature([1], tf.string),
              "img_size": tf.FixedLenFeature([1], tf.string)}
    
  features = tf.parse_single_example(serialized_example, features=features)
  
  image1_buffer = tf.reshape(features["image1_raw"], shape=[])
  image1 = tf.image.decode_jpeg(image1_buffer, channels=COLOR_CHAN)
  image1.set_shape([RESIZE_HEIGHT, RESIZE_WIDTH, COLOR_CHAN])
  image1 = tf.cast(image1, tf.float32) / 255.0
  
  image2_buffer = tf.reshape(features["image2_raw"], shape=[])
  image2 = tf.image.decode_jpeg(image2_buffer, channels=COLOR_CHAN)
  image2.set_shape([RESIZE_HEIGHT, RESIZE_WIDTH, COLOR_CHAN])
  image2 = tf.cast(image2, tf.float32) /255.0
  
  file_name = features['scene']
  img_size = tf.decode_raw(features["img_size"], tf.float32)
  img_size.set_shape([1, 2])
  
  if training:
    image_batch = tf.train.shuffle_batch(
      [image1, image2],
      FLAGS.batch_size,
      num_threads=FLAGS.batch_size,
      capacity=100 * FLAGS.batch_size,
      min_after_dequeue=50 * FLAGS.batch_size,
      enqueue_many=False)
  else:
    image_batch = tf.train.batch(
      [image1, image2, file_name, img_size],
      FLAGS.batch_size / FLAGS.num_gpus,
      #num_threads=FLAGS.batch_size / FLAGS.num_gpus,
      num_threads=1,
      capacity=10 * FLAGS.batch_size,
      #min_after_dequeue=5 * FLAGS.batch_size,
      enqueue_many=False)

  return image_batch

