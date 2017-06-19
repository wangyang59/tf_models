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
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.05),
                      activation_fn=tf.nn.relu):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(images, 32,  [7, 7], stride=2, scope='cnv1')
      cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
      cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
      cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
      cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
      cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
      cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
      cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
      cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
      cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
      cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
      cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
      cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
      cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

      upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
      # There might be dimension mismatch due to uneven down/up-sampling
      upcnv7 = resize_like(upcnv7, cnv6b)
      i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
      icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

      upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
      upcnv6 = resize_like(upcnv6, cnv5b)
      i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
      icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')
      flow6  =  slim.conv2d(icnv6, 2,   [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow6')
      flow6_up = tf.image.resize_bilinear(flow6, [np.int(H/16), np.int(W/16)])

      upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
      upcnv5 = resize_like(upcnv5, cnv4b)
      i5_in  = tf.concat([upcnv5, cnv4b, flow6_up], axis=3)
      icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
      flow5  =  slim.conv2d(icnv5, 2,   [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow5')
      flow5_up = tf.image.resize_bilinear(flow5, [np.int(H/8), np.int(W/8)])
      
      upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
      i4_in  = tf.concat([upcnv4, cnv3b, flow5_up], axis=3)
      icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
      flow4  =  slim.conv2d(icnv4, 2,   [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow4')
      flow4_up = tf.image.resize_bilinear(flow4, [np.int(H/4), np.int(W/4)])

      upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
      i3_in  = tf.concat([upcnv3, cnv2b, flow4_up], axis=3)
      icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
      flow3  = slim.conv2d(icnv3, 2,   [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow3')
      flow3_up = tf.image.resize_bilinear(flow3, [np.int(H/2), np.int(W/2)])

      upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
      i2_in  = tf.concat([upcnv2, cnv1b, flow3_up], axis=3)
      icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
      flow2  = slim.conv2d(icnv2, 2,   [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow2')
      flow2_up = tf.image.resize_bilinear(flow2, [np.int(H), np.int(W)])

      upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
      i1_in  = tf.concat([upcnv1, flow2_up], axis=3)
      icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
      flow1  = slim.conv2d(icnv1, 2,   [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow1')
      
      return flow1, flow2, flow3, flow4, flow5, flow6
