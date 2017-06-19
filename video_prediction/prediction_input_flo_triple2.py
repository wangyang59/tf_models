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

DATA_DIR = '/home/wangyang59/Data/ILSVRC2016_tf_triple'
#DATA_DIR = '/home/wangyang59/Data/ILSVRC2016_tf_stab/train'
FLAGS = flags.FLAGS

# Original image dimensions
ORIGINAL_WIDTH = 256
ORIGINAL_HEIGHT = 256
COLOR_CHAN = 3

# Default image dimensions.
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Dimension of the state and action.
STATE_DIM = 5

def build_tfrecord_input(training=True, blacklist=[]):
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
  filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs=None)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  features = {"image1_raw": tf.FixedLenFeature([1], tf.string),
              "image2_raw": tf.FixedLenFeature([1], tf.string),
              "image3_raw": tf.FixedLenFeature([1], tf.string),
              "file_name": tf.FixedLenFeature([1], tf.string)}
  features = tf.parse_single_example(serialized_example, features=features)
  
  image1_buffer = tf.reshape(features["image1_raw"], shape=[])
  image1 = tf.image.decode_jpeg(image1_buffer, channels=COLOR_CHAN)
  image1.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
  image1 = tf.cast(image1, tf.float32) / 255.0
  
  image2_buffer = tf.reshape(features["image2_raw"], shape=[])
  image2 = tf.image.decode_jpeg(image2_buffer, channels=COLOR_CHAN)
  image2.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
  image2 = tf.cast(image2, tf.float32) / 255.0
  
  image3_buffer = tf.reshape(features["image3_raw"], shape=[])
  image3 = tf.image.decode_jpeg(image3_buffer, channels=COLOR_CHAN)
  image3.set_shape([ORIGINAL_HEIGHT, ORIGINAL_WIDTH, COLOR_CHAN])
  image3 = tf.cast(image3, tf.float32) / 255.0
  
  if IMG_HEIGHT != IMG_WIDTH:
    raise ValueError('Unequal height and width unsupported')
  
  
  image_batch = tf.train.shuffle_batch(
    [image1, image2, image3, features['file_name']],
    FLAGS.batch_size,
    num_threads=FLAGS.batch_size,
    capacity=100 * FLAGS.batch_size,
    min_after_dequeue=1600)

  return image_batch

