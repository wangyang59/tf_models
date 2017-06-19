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
from tensorflow.python.ops import init_ops

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])


def leaky_relu(_x, alpha=0.1):
  pos = tf.nn.relu(_x)
  neg = alpha * (_x - abs(_x)) * 0.5

  return pos + neg

def construct_model(image1, image2, is_training=True):
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

  batch_size, H, W, color_channels = image1.get_shape()[0:4]
  images = tf.concat([image1, image2], axis=3)
  
  #############################
  
  batch_norm_params = {'is_training': is_training}

  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(images, 64,  [7, 7], stride=2, scope='cnv1')
      cnv2  = slim.conv2d(cnv1, 128,  [5, 5], stride=2, scope='cnv2')
      cnv3  = slim.conv2d(cnv2, 256, [5, 5], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3,  256, [3, 3], stride=1, scope='cnv3b')
      cnv4  = slim.conv2d(cnv3b, 512, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4,  512, [3, 3], stride=1, scope='cnv4b')
      cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
      cnv6  = slim.conv2d(cnv5b, 1024, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6,  1024, [3, 3], stride=1, scope='cnv6b')
      
      flow6  =  slim.conv2d(cnv6b, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow6')
      
      deconv5 = slim.conv2d_transpose(cnv6b, 512, [4, 4], stride=2, scope='deconv5', weights_regularizer=None)
      flow6to5 = slim.conv2d_transpose(flow6, 2, [4, 4], stride=2, scope='flow6to5', weights_regularizer=None)
      concat5 = tf.concat([cnv5b, deconv5, flow6to5], axis=3)
      flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow5')
      
      deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], stride=2, scope='deconv4', weights_regularizer=None)
      flow5to4 = slim.conv2d_transpose(flow5, 2, [4, 4], stride=2, scope='flow5to4', weights_regularizer=None)
      concat4 = tf.concat([cnv4b, deconv4, flow5to4], axis=3)
      flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow4')
      
      deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], stride=2, scope='deconv3', weights_regularizer=None)
      flow4to3 = slim.conv2d_transpose(flow4, 2, [4, 4], stride=2, scope='flow4to3', weights_regularizer=None)
      concat3 = tf.concat([cnv3b, deconv3, flow4to3], axis=3)
      flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow3')
      
      deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], stride=2, scope='deconv2', weights_regularizer=None)
      flow3to2 = slim.conv2d_transpose(flow3, 2, [4, 4], stride=2, scope='flow3to2', weights_regularizer=None)
      concat2 = tf.concat([cnv2, deconv2, flow3to2], axis=3)
      flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow2')
      
      return flow2, flow3, flow4, flow5, flow6
