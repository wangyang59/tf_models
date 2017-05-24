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

"""Model architecture for predictive model, including CDNA, DNA, and STP."""
"""use directly from flow data"""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from lstm_ops import basic_conv_lstm_cell
from tensorflow.python.ops import init_ops


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5

def sample_gumbel(shape, eps=1e-20): 
  """Sample from Gumbel(0, 1)"""
  U = tf.random_uniform(shape,minval=0,maxval=1)
  return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature): 
  """ Draw a sample from the Gumbel-Softmax distribution"""
  y = logits + sample_gumbel(tf.shape(logits))
  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):
  """Sample from the Gumbel-Softmax distribution and optionally discretize.
  Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
  Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
  """
  y = gumbel_softmax_sample(logits, temperature)
  if hard:
    #k = tf.shape(logits)[-1]
    #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y
  return y

def construct_model(image):
  """Build convolutional lstm video predictor using STP, CDNA, or DNA.

  Args:
    images: tensor of ground truth image sequences
    actions: tensor of action sequences
    states: tensor of ground truth state sequences
    iter_num: tensor of the current training iteration (for sched. sampling)
    k: constant used for scheduled sampling. -1 to feed in own prediction.
    use_state: True to include state and action in prediction
    num_masks: the number of different pixel motion predictions (and
               the number of masks for each of those predictions)
    stp: True to use Spatial Transformer Predictor (STP)
    cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
    dna: True to use Dynamic Neural Advection (DNA)
    context_frames: number of ground truth frames to pass in before
                    feeding in own predictions
  Returns:
    gen_images: predicted future image frames
    gen_states: predicted future states

  Raises:
    ValueError: if more than one network option specified or more than 1 mask
    specified for DNA model.
  """

  batch_size, img_height, img_width, color_channels = image.get_shape()[0:4]
  lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))

  #############################

  enc0_s = slim.layers.conv2d(
      image,
      32, [5, 5],
      stride=2,
      scope='scale1_conv1_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm1_s'})
  
  hidden1_s = slim.layers.conv2d(
      enc0_s,
      lstm_size[0], [5, 5],
      stride=1,
      scope='state1_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm2_s'})
  
  hidden2_s = slim.layers.conv2d(
      hidden1_s,
      lstm_size[1], [5, 5],
      stride=1,
      scope='state2_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm3_s'})

  enc1_s = slim.layers.conv2d(
      hidden2_s, hidden2_s.get_shape()[3], [3, 3], stride=2, scope='conv2_s')
  
  hidden3_s = slim.layers.conv2d(
      enc1_s,
      lstm_size[2], [5, 5],
      stride=1,
      scope='state3_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm4_s'})
  
  hidden4_s = slim.layers.conv2d(
      hidden3_s,
      lstm_size[3], [5, 5],
      stride=1,
      scope='state4_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm5_s'})
  
  enc2_s = slim.layers.conv2d(
      hidden4_s, hidden4_s.get_shape()[3], [3, 3], stride=2, scope='conv3_s')

  enc3_s = slim.layers.conv2d(
      enc2_s, hidden4_s.get_shape()[3], [1, 1], stride=1, scope='conv4_s')
  
  hidden5_s = slim.layers.conv2d(
      enc3_s,
      lstm_size[4], [5, 5],
      stride=1,
      scope='state5_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm6_s'})
  
  enc4_s = slim.layers.conv2d_transpose(
      hidden5_s, hidden5_s.get_shape()[3], 4, stride=2, scope='convt1_s')
  
  hidden6_s = slim.layers.conv2d(
      enc4_s,
      lstm_size[5], [5, 5],
      stride=1,
      scope='state6_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm7_s'})
  
  # Skip connection.
  hidden6_s = tf.concat(axis=3, values=[hidden6_s, enc1_s])  # both 16x16

  enc5_s = slim.layers.conv2d_transpose(
      hidden6_s, hidden6_s.get_shape()[3], 4, stride=2, scope='convt2_s')
  
  hidden7_s = slim.layers.conv2d(
      enc5_s,
      lstm_size[6], [5, 5],
      stride=1,
      scope='state7_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm8_s'})

  # Skip connection.
  hidden7_s= tf.concat(axis=3, values=[hidden7_s, enc0_s])  # both 32x32

  enc6_s = slim.layers.conv2d_transpose(
      hidden7_s,
      hidden7_s.get_shape()[3], 4, stride=2, scope='convt3_s',
      normalizer_fn=tf_layers.layer_norm,
      normalizer_params={'scope': 'layer_norm9_s'})
  
  masks_s = slim.layers.conv2d_transpose(
      enc6_s, 2, 1, stride=1, scope='convt7_s')
  masks_probs_s = tf.nn.softmax(tf.reshape(masks_s, [-1, 2]))
  #entropy_losses.append(tf.reduce_mean(-tf.reduce_sum(masks_probs * tf.log(masks_probs + 1e-10), [1])))
  masks_s = tf.reshape(
      masks_probs_s,
      #gumbel_softmax(tf.reshape(masks, [-1, num_masks]), TEMP, hard=False),
      [int(batch_size), int(img_height), int(img_width), 2])
  poss_move_mask, bg_mask = tf.split(axis=3, num_or_size_splits=2, value=masks_s)

  return poss_move_mask, bg_mask

