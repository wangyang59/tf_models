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
import random

DATA_DIR = '/home/wangyang59/Data/ILSVRC2016_tf_chair'
#DATA_DIR = '/home/wangyang59/Data/ILSVRC2016_tf_stab/train'
FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 512
ORIGINAL_HEIGHT = 384
COLOR_CHAN = 3


def build_tfrecord_input(training=True, blacklist=[], num_epochs=None):
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
  filenames = gfile.Glob(os.path.join(FLAGS.data_dir, '*'))
  filenames = filter(lambda x: x.split("/")[-1] not in blacklist, filenames)
  if not filenames:
    raise RuntimeError('No data files found.')
  index = int(np.floor(FLAGS.train_val_split * len(filenames)))
  if training:
    filenames = filenames[:index]
  else:
    filenames = filenames[index:]
  filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=num_epochs)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = {"image1_raw": tf.FixedLenFeature([1], tf.string),
              "image2_raw": tf.FixedLenFeature([1], tf.string),
              "flo": tf.FixedLenFeature([1], tf.string)}
  features = tf.parse_single_example(serialized_example, features=features)
  
  image1_buffer = tf.reshape(features["image1_raw"], shape=[])
  image1 = tf.image.decode_jpeg(image1_buffer, channels=COLOR_CHAN)
  image1.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
  image1 = tf.cast(image1, tf.float32) / 255.0
  
  image2_buffer = tf.reshape(features["image2_raw"], shape=[])
  image2 = tf.image.decode_jpeg(image2_buffer, channels=COLOR_CHAN)
  image2.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
  image2 = tf.cast(image2, tf.float32) / 255.0
  
  flo = tf.decode_raw(features['flo'], tf.float32)
  flo = tf.reshape(flo, [ORIGINAL_HEIGHT, ORIGINAL_WIDTH, 2])
  
  if training:
    images = tf.concat([image1, image2], axis=2)
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.cond(tf.random_uniform([]) < 0.5, lambda: tf.image.rot90(images, 2), lambda: images)
    images. set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN*2])
    image1, image2 =  tf.split(axis=2, num_or_size_splits=2, value=images)
#   if random.random() < 0.5:
#     image1 = tf.image.flip_left_right(image1)
#     image2 = tf.image.flip_left_right(image2)
#   
#   if random.random() < 0.5:
#     image1 = tf.image.flip_up_down(image1)
#     image2 = tf.image.flip_up_down(image2)
  
#   if random.random() < 0.5:
#     image1 = tf.image.rot90(image1, 2)
#     image2 = tf.image.rot90(image2, 2)
  
#   brightness = random.gauss(0, 0.2)
#   image1 = tf.clip_by_value(image1+brightness, 0.0, 1.0)
#   image2 = tf.clip_by_value(image2+brightness, 0.0, 1.0)
  
#   contrast = random.random()*1.6 + 0.2
#   image1 = tf.image.adjust_contrast(image1, contrast)
#   image2 = tf.image.adjust_contrast(image2, contrast)
#   
#   gamma = random.random()*0.6 + 0.7
#   image1 = tf.image.adjust_gamma(image1, gamma)
#   image2 = tf.image.adjust_gamma(image2, gamma)
  
  if training:
    image_batch = tf.train.shuffle_batch(
      [image1, image2, flo],
      FLAGS.batch_size,
      num_threads=FLAGS.batch_size,
      capacity=100 * FLAGS.batch_size,
      min_after_dequeue=50 * FLAGS.batch_size,
      enqueue_many=False)
  else:
    image_batch = tf.train.batch(
      [image1, image2, flo],
      FLAGS.batch_size / FLAGS.num_gpus,
      #num_threads=FLAGS.batch_size / FLAGS.num_gpus,
      num_threads=1,
      capacity=10 * FLAGS.batch_size,
      #min_after_dequeue=5 * FLAGS.batch_size,
      enqueue_many=False)

  return image_batch

