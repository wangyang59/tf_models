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
from optical_flow_warp import transformer

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

def autoencoder(image, reuse_scope=False, trainable=True):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      reuse=reuse_scope,
                      trainable=trainable,
                      variables_collections=["ae"]):
    cnv1  = slim.conv2d(image, 16,  [3, 3], stride=2, scope='cnv1_ae')
    cnv2  = slim.conv2d(cnv1, 32,  [3, 3], stride=2, scope='cnv2_ae')
    cnv3  = slim.conv2d(cnv2, 64, [3, 3], stride=2, scope='cnv3_ae')
    cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4_ae')
    cnv5  = slim.conv2d(cnv4, 128, [3, 3], stride=2, scope='cnv5_ae')
    cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6_ae')
    
    deconv5 = slim.conv2d_transpose(cnv6, 128, [4, 4], stride=2, scope='deconv5_ae')
    deconv4 = slim.conv2d_transpose(deconv5, 128, [4, 4], stride=2, scope='deconv4_ae')
    deconv3 = slim.conv2d_transpose(deconv4, 64, [4, 4], stride=2, scope='deconv3_ae')
    deconv2 = slim.conv2d_transpose(deconv3, 32, [4, 4], stride=2, scope='deconv2_ae')
    deconv1 = slim.conv2d_transpose(deconv2, 16, [4, 4], stride=2, scope='deconv1_ae')
    recon   = slim.conv2d_transpose(deconv1, 3, [4, 4], stride=2, scope='recon_ae', activation_fn=tf.nn.sigmoid)
    
    return recon, [cnv2, cnv3, cnv4, cnv5, cnv6]

def decoder(feature, reuse_scope=True, trainable=True, level=None):
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      reuse=reuse_scope,
                      trainable=trainable,
                      variables_collections=["ae"]):
    if level == 5:
      feature = slim.conv2d(feature, 256, [3, 3], stride=2, scope='cnv6_ae')
    deconv5 = slim.conv2d_transpose(feature, 128, [4, 4], stride=2, scope='deconv5_ae')
    deconv4 = slim.conv2d_transpose(deconv5, 128, [4, 4], stride=2, scope='deconv4_ae')
    deconv3 = slim.conv2d_transpose(deconv4, 64, [4, 4], stride=2, scope='deconv3_ae')
    deconv2 = slim.conv2d_transpose(deconv3, 32, [4, 4], stride=2, scope='deconv2_ae')
    deconv1 = slim.conv2d_transpose(deconv2, 16, [4, 4], stride=2, scope='deconv1_ae')
    recon   = slim.conv2d_transpose(deconv1, 3, [4, 4], stride=2, scope='recon_ae', activation_fn=tf.nn.sigmoid)

    return recon
  
def sub_model(inputs, level):
#   batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
#   inputs = tf.concat([image1, image1_warp, flo], axis=3)
#   inputs.set_shape([batch_size, H, W, 8])
  #############################
  
  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      #weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=tf.nn.relu):
                      #outputs_collections=end_points_collection):
      cnv1  = slim.conv2d(inputs, 32,  [7, 7], stride=1, scope='cnv1_'+str(level))
      cnv2  = slim.conv2d(cnv1, 64,  [7, 7], stride=1, scope='cnv2_'+str(level))
      cnv3  = slim.conv2d(cnv2, 32, [7, 7], stride=1, scope='cnv3_'+str(level))
      cnv4 = slim.conv2d(cnv3,  16, [7, 7], stride=1, scope='cnv4_'+str(level))
      cnv5  = slim.conv2d(cnv4, 2, [7, 7], stride=1, scope='cnv5_'+str(level))
      
      return cnv5

def construct_model(image1, image2, image1_pyrimad, image2_pyrimad, is_training=True):
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

  batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
  images = tf.concat([image1, image2], axis=3)
  
  image1_2, image1_3, image1_4, image1_5, image1_6 = image1_pyrimad
  image2_2, image2_3, image2_4, image2_5, image2_6 = image2_pyrimad
  
  #############################
  
  batch_norm_params = {'is_training': is_training}

  with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                      #normalizer_fn=slim.batch_norm,
                      #normalizer_params=batch_norm_params,
                      weights_regularizer=slim.l2_regularizer(0.0004),
                      activation_fn=leaky_relu,
                      variables_collections=["flownet"]):
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
      image1_6p = transformer(image2_6, 20*flow6/64.0, [H/64, W/64])
      
      deconv5 = slim.conv2d_transpose(cnv6b, 512, [4, 4], stride=2, scope='deconv5', weights_regularizer=None)
      flow6to5 = tf.image.resize_bilinear(flow6, [H/(2**5), (W/(2**5))])
      feature6to5 = tf.image.resize_bilinear(tf.concat([image1_6, image2_6, image1_6p, image1_6-image1_6p], axis=3), [H/(2**5), W/(2**5)])
      feature6to5.set_shape([batch_size, H/(2**5), W/(2**5), color_channels*4])
      
      concat5 = tf.concat([cnv5b, deconv5, sub_model(feature6to5, level=5)], axis=3)
      flow5 = slim.conv2d(concat5, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow5') + flow6to5
      image1_5p = transformer(image2_5, 20*flow5/32.0, [H/32, W/32])
      
      deconv4 = slim.conv2d_transpose(concat5, 256, [4, 4], stride=2, scope='deconv4', weights_regularizer=None)
      flow5to4 = tf.image.resize_bilinear(flow5, [H/(2**4), (W/(2**4))])
      feature5to4 = tf.image.resize_bilinear(tf.concat([image1_5, image2_5, image1_5p, image1_5-image1_5p], axis=3), [H/(2**4), (W/(2**4))])
      feature5to4.set_shape([batch_size, H/(2**4), W/(2**4), color_channels*4])
      
      concat4 = tf.concat([cnv4b, deconv4, sub_model(feature5to4, level=4)], axis=3)
      flow4 = slim.conv2d(concat4, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow4') + flow5to4
      image1_4p = transformer(image2_4, 20*flow4/16.0, [H/16, W/16])
      
      deconv3 = slim.conv2d_transpose(concat4, 128, [4, 4], stride=2, scope='deconv3', weights_regularizer=None)
      flow4to3 = tf.image.resize_bilinear(flow4, [H/(2**3), (W/(2**3))])
      feature4to3 = tf.image.resize_bilinear(tf.concat([image1_4, image2_4, image1_4p, image1_4-image1_4p], axis=3), [H/(2**3), (W/(2**3))])
      feature4to3.set_shape([batch_size, H/(2**3), W/(2**3), color_channels*4])
      
      concat3 = tf.concat([cnv3b, deconv3, sub_model(feature4to3, level=3)], axis=3)
      flow3 = slim.conv2d(concat3, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow3') + flow4to3
      image1_3p = transformer(image2_3, 20*flow3/8.0, [H/8, W/8])

      deconv2 = slim.conv2d_transpose(concat3, 64, [4, 4], stride=2, scope='deconv2', weights_regularizer=None)
      flow3to2 = tf.image.resize_bilinear(flow3, [H/(2**2), (W/(2**2))])
      feature3to2 = tf.image.resize_bilinear(tf.concat([image1_3, image2_3, image1_3p, image1_3-image1_3p], axis=3), [H/(2**2), (W/(2**2))])
      feature3to2.set_shape([batch_size, H/(2**2), W/(2**2), color_channels*4])
      
      concat2 = tf.concat([cnv2, deconv2, sub_model(feature3to2, level=2)], axis=3)
      flow2 = slim.conv2d(concat2, 2, [3, 3], stride=1, 
          activation_fn=None, normalizer_fn=None, scope='flow2') + flow3to2
      
      image1_2p = transformer(image2_2, 20*flow2/4.0, [H/4, W/4])
      
      return flow2, flow3, flow4, flow5, flow6, [image1_2p, image1_3p, image1_4p, image1_5p, image1_6p]
