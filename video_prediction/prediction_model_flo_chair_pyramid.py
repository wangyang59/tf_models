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
from tensorflow.contrib.labeled_tensor import batch

"""Model architecture for predictive model, including CDNA, DNA, and STP."""
"""use directly from flow data"""

import tensorflow as tf

import tensorflow.contrib.slim as slim
from optical_flow_warp import transformer

def construct_model(image1, image2, flo, level):
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
  image1_warp = transformer(image2, 20*flo/(2**level), [H, W])
  inputs = tf.concat([image1, image1_warp, flo], axis=3)
  inputs.set_shape([batch_size, H, W, 8])
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
      cnv5  = slim.conv2d(cnv4, 2, [7, 7], stride=1, scope='cnv5_'+str(level), activation_fn=None)
      
      return cnv5
