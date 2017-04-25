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

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from lstm_ops import basic_conv_lstm_cell

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

def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    use_state=True,
                    num_masks=10,
                    stp=False,
                    cdna=True,
                    dna=False,
                    context_frames=2,
                    global_shift=0):
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

  batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
  lstm_func = basic_conv_lstm_cell

  # Generated robot states and images.
  gen_states, gen_images = [], []
  current_state = states[0]
  shifted_masks = []
  mask_lists = []
  entropy_losses = []
  gs_kernels = []
  
  if k == -1:
    feedself = True
  else:
    # Scheduled sampling:
    # Calculate number of ground-truth frames to pass in.
    num_ground_truth = tf.to_int32(
        tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
    feedself = False

  # LSTM state sizes and states.
  lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
  lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
  lstm_state5, lstm_state6, lstm_state7 = None, None, None

  for image, action in zip(images[:-1], actions[:-1]):
    # Reuse variables after the first timestep.
    reuse = bool(gen_images)

    done_warm_start = len(gen_images) > context_frames - 1
    with slim.arg_scope(
        [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
         tf_layers.layer_norm, slim.layers.conv2d_transpose],
        reuse=reuse):
      
      if k == 0:
        prev_image = image
      else:
        if feedself and done_warm_start:
          # Feed in generated image.
          prev_image = gen_images[-1]
        elif done_warm_start:
          # Scheduled sampling
          prev_image = scheduled_sample(image, gen_images[-1], batch_size,
                                        num_ground_truth)
        else:
          # Always feed in ground_truth
          prev_image = image

      # Predicted state is always fed back in
      state_action = tf.concat(axis=1, values=[action, current_state])

      enc0 = slim.layers.conv2d(
          prev_image,
          32, [5, 5],
          stride=2,
          scope='scale1_conv1',
          normalizer_fn=tf_layers.layer_norm,
          normalizer_params={'scope': 'layer_norm1'})

      hidden1, lstm_state1 = lstm_func(
          enc0, lstm_state1, lstm_size[0], scope='state1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
      hidden2, lstm_state2 = lstm_func(
          hidden1, lstm_state2, lstm_size[1], scope='state2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
      enc1 = slim.layers.conv2d(
          hidden2, hidden2.get_shape()[3], [3, 3], stride=2, scope='conv2')

      hidden3, lstm_state3 = lstm_func(
          enc1, lstm_state3, lstm_size[2], scope='state3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
      hidden4, lstm_state4 = lstm_func(
          hidden3, lstm_state4, lstm_size[3], scope='state4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm5')
      enc2 = slim.layers.conv2d(
          hidden4, hidden4.get_shape()[3], [3, 3], stride=2, scope='conv3')

      # Pass in state and action.
      smear = tf.reshape(
          state_action,
          [int(batch_size), 1, 1, int(state_action.get_shape()[1])])
      smear = tf.tile(
          smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
      if use_state:
        enc2 = tf.concat(axis=3, values=[enc2, smear])
      enc3 = slim.layers.conv2d(
          enc2, hidden4.get_shape()[3], [1, 1], stride=1, scope='conv4')

      hidden5, lstm_state5 = lstm_func(
          enc3, lstm_state5, lstm_size[4], scope='state5')  # last 8x8
      hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
      enc4 = slim.layers.conv2d_transpose(
          hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

      hidden6, lstm_state6 = lstm_func(
          enc4, lstm_state6, lstm_size[5], scope='state6')  # 16x16
      hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')
      # Skip connection.
      hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

      enc5 = slim.layers.conv2d_transpose(
          hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
      hidden7, lstm_state7 = lstm_func(
          enc5, lstm_state7, lstm_size[6], scope='state7')  # 32x32
      hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

      # Skip connection.
      hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

      enc6 = slim.layers.conv2d_transpose(
          hidden7,
          hidden7.get_shape()[3], 3, stride=2, scope='convt3',
          normalizer_fn=tf_layers.layer_norm,
          normalizer_params={'scope': 'layer_norm9'})


      # Using largest hidden state for predicting a new image layer.
      enc7 = slim.layers.conv2d_transpose(
          enc6, color_channels, 1, stride=1, scope='convt4')
      # This allows the network to also generate one image from scratch,
      # which is useful when regions of the image become unoccluded.
      guessed = tf.nn.sigmoid(enc7)
      
      if global_shift:
        cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
        prev_image, gs_kernel = do_global_shift(prev_image, cdna_input, int(color_channels))
        gs_kernels.append(gs_kernel)
      
      masks = slim.layers.conv2d_transpose(
          enc6, num_masks, 1, stride=1, scope='convt7')
      masks_probs = tf.nn.softmax(tf.reshape(masks, [-1, num_masks]))
      entropy_losses.append(tf.reduce_mean(-tf.reduce_sum(masks_probs * tf.log(masks_probs + 1e-10), [1])))
      masks = tf.reshape(
          masks_probs,
          #gumbel_softmax(tf.reshape(masks, [-1, num_masks]), TEMP, hard=False),
          [int(batch_size), int(img_height), int(img_width), num_masks])
      mask_list = tf.split(axis=3, num_or_size_splits=num_masks, value=masks)
      
      mask_lists.append(mask_list)
      
      output, shifted_mask = my_transformation2(prev_image, mask_list, int(color_channels), guessed)
        
      gen_images.append(output)
      shifted_masks.append(shifted_mask)

      current_state = slim.layers.fully_connected(
          state_action,
          int(current_state.get_shape()[1]),
          scope='state_pred',
          activation_fn=None)
      gen_states.append(current_state)

  return gen_images, gen_states, shifted_masks, mask_lists, entropy_losses, gs_kernels


# def my_transformation(prev_image, mask_list, color_channels, guessed):
#   kernels = []
#   for i in xrange(DNA_KERN_SIZE * DNA_KERN_SIZE):
#     if i != DNA_KERN_SIZE * DNA_KERN_SIZE / 2:
#       kernel = np.zeros((DNA_KERN_SIZE * DNA_KERN_SIZE), dtype=np.float32)
#       kernel[i] = 1.0
#       kernel = kernel.reshape((DNA_KERN_SIZE, DNA_KERN_SIZE, 1, 1))
#       kernel = tf.constant(kernel, shape=(DNA_KERN_SIZE, DNA_KERN_SIZE, 1, 1), 
#                            name='kernel'+str(i), verify_shape=True)
#       kernels.append(tf.tile(kernel, [1, 1, color_channels, 1]))
#   
#   assert len(kernels) == len(mask_list) - 2
#   
#   output = prev_image * mask_list[0] + guessed * mask_list[1]
#   for kernel, mask in zip(kernels, mask_list[2:]):
#     output += tf.nn.depthwise_conv2d(prev_image * mask, kernel, [1, 1, 1, 1], 'SAME')
#   
#   return output

def do_global_shift(prev_image, cdna_input, color_channels):
  batch_size = int(cdna_input.get_shape()[0])

  # Predict kernels using linear function of last hidden layer.
  cdna_kerns = slim.layers.fully_connected(
      cdna_input,
      DNA_KERN_SIZE * DNA_KERN_SIZE,
      scope='global_shift_params',
      activation_fn=None)

  # Reshape and normalize.
  cdna_kerns = tf.reshape(
      cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, 1])
  cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
  norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
  cdna_kerns /= norm_factor
  cdna_kerns_output = cdna_kerns

  cdna_kerns = tf.tile(cdna_kerns, [1, 1, 1, color_channels, 1])
  cdna_kerns = tf.split(axis=0, num_or_size_splits=batch_size, value=cdna_kerns)
  prev_images = tf.split(axis=0, num_or_size_splits=batch_size, value=prev_image)

  # Transform image.
  transformed = []
  for kernel, preimg in zip(cdna_kerns, prev_images):
    kernel = tf.squeeze(kernel)
    if len(kernel.get_shape()) == 3:
      kernel = tf.expand_dims(kernel, -1)
    transformed.append(
        tf.nn.depthwise_conv2d(preimg, kernel, [1, 1, 1, 1], 'SAME'))
  transformed = tf.concat(axis=0, values=transformed)
  return transformed, tf.squeeze(cdna_kerns_output)

def my_transformation2(prev_image, mask_list, color_channels, guessed):
  kernels = []
  for i in xrange(DNA_KERN_SIZE * DNA_KERN_SIZE):
    if i != DNA_KERN_SIZE * DNA_KERN_SIZE / 2:
      kernel = np.zeros((DNA_KERN_SIZE * DNA_KERN_SIZE), dtype=np.float32)
      kernel[i] = 1.0
      kernel = kernel.reshape((DNA_KERN_SIZE, DNA_KERN_SIZE, 1, 1))
      kernel = tf.constant(kernel, shape=(DNA_KERN_SIZE, DNA_KERN_SIZE, 1, 1), 
                           name='kernel'+str(i), verify_shape=True)
      kernels.append(kernel)
  
  assert len(kernels) == len(mask_list) - 2
  
  # mask[0] indicates stay, mask[1] indicates disappear
  output = prev_image * mask_list[0]
  shifted_mask = mask_list[0] + mask_list[1]
  for kernel, mask in zip(kernels, mask_list[2:]):
    tmp_mask = tf.nn.depthwise_conv2d(mask, kernel, [1, 1, 1, 1], 'SAME')
    output += tmp_mask * tf.nn.depthwise_conv2d(prev_image, tf.tile(kernel, [1, 1, color_channels, 1]), 
                                     [1, 1, 1, 1], 'SAME')
    shifted_mask += tmp_mask
  
  output += guessed * tf.nn.relu(1.0 - shifted_mask)
  return output, shifted_mask

def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
  """Sample batch with specified mix of ground truth and generated data points.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    num_ground_truth: number of ground-truth examples to include in batch.
  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  idx = tf.random_shuffle(tf.range(int(batch_size)))
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)
  return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                           [ground_truth_examps, generated_examps])