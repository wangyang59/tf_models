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
from tensorflow.examples.tutorials.mnist.mnist import loss

"""Code for training the prediction model."""

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from prediction_input_flo_triple2 import build_tfrecord_input, DATA_DIR
from prediction_model_flo_learn import construct_model
from visualize import plot_flo_triple
from optical_flow_warp import transformer

import os

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 500

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', "", 'directory for model checkpoints.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_float('train_val_split', 1.0,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')
flags.DEFINE_integer('num_gpus', 1,
                   'the number of gpu to use')


def get_black_list(clses):
  blacklist = []
  for cls in clses:
    fname = "/home/wangyang59/Data/ILSVRC2016/ImageSets/VID/train_%s.txt" % cls
    with open(fname) as f:
      content = f.readlines()
    blacklist += [x.split(" ")[0].split("/")[-1] + ".tfrecord" for x in content]
  return blacklist

## Helper functions
def peak_signal_to_noise_ratio(true, pred):
  """Image quality metric based on maximal signal power vs. power of the noise.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    peak signal to noise ratio (PSNR)
  """
  return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))

def weighted_mean_squared_error(true, pred, weight):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  
  tmp = tf.reduce_sum(weight*tf.square(true-pred), axis=[1,2], keep_dims=True) / tf.reduce_sum(weight, axis=[1, 2], keep_dims=True)
  return tf.reduce_mean(tmp)
  #return tf.reduce_sum(tf.square(true - pred)*weight) / tf.to_float(tf.size(pred))
  #return tf.reduce_sum(tf.square(true - pred)*weight) / tf.reduce_sum(weight)

def mean_L1_error(true, pred):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.abs(true - pred)) / tf.to_float(tf.size(pred))

def weighted_mean_L1_error(true, pred, weight):
  """L2 distance between tensors true and pred.

  Args:
    true: the ground truth image.
    pred: the predicted image.
  Returns:
    mean squared error between ground truth and predicted image.
  """
  return tf.reduce_sum(tf.abs(true - pred)*weight) / tf.to_float(tf.size(pred))

def huber_error(true, pred, delta=0.05):
  err = true - pred 
  herr = tf.where(tf.abs(err)<delta, 0.5*tf.square(err), delta*(tf.abs(err) - 0.5*delta)) # condition, true, false
  return tf.reduce_sum(herr) / tf.to_float(tf.size(pred))

def cal_grad_error_wmask(image, mask, kernels):
  """Calculate the gradient of the given image by calculate the difference between nearby pixels
  """
  error = 0.0
  img_height, img_width, color_channels = map(int, image.get_shape()[1:4])
  
  cropped_image = image[:, 1:(img_height-1), 1:(img_width-1), :]
  cropped_mask = mask[:, 1:(img_height-1), 1:(img_width-1), :]
  for kernel in kernels:
    shifted_image = tf.nn.depthwise_conv2d(image, tf.tile(kernel, [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
    error += weighted_mean_squared_error(cropped_image, shifted_image[:, 1:(img_height-1), 1:(img_width-1), :], cropped_mask)
    
  return error / len(kernels)
  
def cal_grad_error(image, kernels):
  """Calculate the gradient of the given image by calculate the difference between nearby pixels
  """
  error = 0.0
  img_height, img_width, color_channels = map(int, image.get_shape()[1:4])
  cropped_image = image[:, 1:(img_height-1), 1:(img_width-1), :]
  for kernel in kernels:
    shifted_image = tf.nn.depthwise_conv2d(image, tf.tile(kernel, [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
    error += mean_squared_error(cropped_image, shifted_image[:, 1:(img_height-1), 1:(img_width-1), :])
    
  return error / len(kernels)

def cal_weighted_var(image, mask):
  
  weighted_mean = tf.reduce_sum(image*mask, axis=[1, 2], keep_dims=True) / tf.reduce_sum(mask, axis=[1, 2], keep_dims=True)
  #mean = tf.reduce_mean(image, axis=[1, 2], keep_dims=True)
  weighted_var = (tf.reduce_sum(mask*tf.square(image - weighted_mean), axis=[1,2], keep_dims=True) + 0.0) / tf.reduce_sum(mask, axis=[1, 2], keep_dims=True)
  
  return tf.reduce_mean(weighted_var)
  
def cal_weighted_edge_diff(image, mask, kernels, scale=1.0):
  error =  0.0
  img_height, img_width, color_channels = map(int, image.get_shape()[1:4])
  cropped_image = image[:, 1:(img_height-1), 1:(img_width-1), :]
  cropped_mask = mask[:, 1:(img_height-1), 1:(img_width-1), :]
  
  for kernel in kernels:
    shifted_image = (tf.nn.depthwise_conv2d(image, tf.tile(kernel, [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME'))[:, 1:(img_height-1), 1:(img_width-1), :]
    shifted_mask = (tf.nn.depthwise_conv2d(mask, tf.tile(kernel, [1, 1, 1, 1]), 
                                           [1, 1, 1, 1], 'SAME'))[:, 1:(img_height-1), 1:(img_width-1), :]
    
    tmp = tf.exp(-tf.reduce_sum(tf.square(shifted_image-cropped_image), axis=[3], keep_dims=True) / scale) * tf.square(shifted_mask-cropped_mask)
    
    error += tf.reduce_sum(tmp) / tf.to_float(tf.size(shifted_mask))
    
  return error / len(kernels)

def get_grad_image(image):
  batch_size, img_height, img_width, color_channels = map(int, image.get_shape()[0:4])
  
  kernels = []
  for i in xrange(4):
    kernel = np.zeros((3 * 3), dtype=np.float32)
    kernel[i*2 + 1] = 1.0
    kernel = kernel.reshape((3, 3, 1, 1))
    kernel = tf.constant(kernel, shape=(3, 3, 1, 1), 
                         name='kernel_shift'+str(i), verify_shape=True)
    kernels.append(kernel)
  
  image_shift_0 = tf.nn.depthwise_conv2d(image, tf.tile(kernels[0], [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
  image_shift_1 = tf.nn.depthwise_conv2d(image, tf.tile(kernels[1], [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
  image_shift_2 = tf.nn.depthwise_conv2d(image, tf.tile(kernels[2], [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
  image_shift_3 = tf.nn.depthwise_conv2d(image, tf.tile(kernels[3], [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
  
  return tf.concat([image, image-image_shift_0, image-image_shift_1, 
                    image-image_shift_2, image-image_shift_3], axis=3)
  
  
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

class Model(object):

  def __init__(self,
               image1=None,
               image2=None,
               image3=None,
               file_names=None,
               reuse_scope=False,
               scope=None,
               prefix="train"):

    #self.prefix = prefix = tf.placeholder(tf.string, [])
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []
    self.kernels = []
    for i in xrange(4):
      kernel = np.zeros((3 * 3), dtype=np.float32)
      kernel[i*2 + 1] = 1.0
      kernel = kernel.reshape((3, 3, 1, 1))
      kernel = tf.constant(kernel, shape=(3, 3, 1, 1),
                           name='kernel_shift'+str(i), verify_shape=True)
      self.kernels.append(kernel)
    
    batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
    
    if not reuse_scope:
      flow1, flow2, flow3, flow4 = construct_model(image2, image3)
    else:  # If it's a validation or test model.
      with tf.variable_scope(scope, reuse=True):
        flow1, flow2, flow3, flow4 = construct_model(image2, image3)
        
    flow4t = transformer(flow4, flow4/8.0, [H/8, W/8])
    flow3t = transformer(flow3, flow3/4.0, [H/4, W/4])
    flow2t = transformer(flow2, flow2/2.0, [H/2, W/2])
    flow1t = transformer(flow1, flow1/1.0, [H, W])
    
    image2g = get_grad_image(image2)
    image1g = get_grad_image(image1)
    
    image1g_4 = transformer(tf.image.resize_bilinear(image2g, [H/8, W/8]), flow4t/8.0, [H/8, W/8])
    loss4 = mean_squared_error(tf.image.resize_bilinear(image1g, [H/8, W/8]), image1g_4)
    
    image1g_3 = transformer(tf.image.resize_bilinear(image2g, [H/4, W/4]), flow3t/4.0, [H/4, W/4])
    loss3 = mean_squared_error(tf.image.resize_bilinear(image1g, [H/4, W/4]), image1g_3)
    
    image1g_2 = transformer(tf.image.resize_bilinear(image2g, [H/2, W/2]), flow2t/2.0, [H/2, W/2])
    loss2 = mean_squared_error(tf.image.resize_bilinear(image1g, [H/2, W/2]), image1g_2)
    
    image1g_1 = transformer(image2g, flow1t, [H, W])
    loss1 = mean_squared_error(image1g, image1g_1)
    
    grad_error4 = cal_grad_error(flow4, self.kernels)
    grad_error3 = cal_grad_error(flow3, self.kernels)
    grad_error2 = cal_grad_error(flow2, self.kernels)
    grad_error1 = cal_grad_error(flow1, self.kernels)
    
    loss = loss1 + loss2 + loss3 + loss4 + (grad_error1 + grad_error2 + grad_error3 + grad_error4)*1.0
        
    self.loss = loss
    self.orig_image1 = image1
    self.orig_image2 = image2
    self.orig_image3 = image3
    self.pred_flo = flow1
    self.pred_warp = image1g_1[:, :, :, 0:3]
    self.file_names = file_names
        
    summaries.append(tf.summary.scalar(prefix + '_loss', self.loss))
    summaries.append(tf.summary.scalar(prefix + '_loss1', loss1))
    summaries.append(tf.summary.scalar(prefix + '_loss2', loss2))
    summaries.append(tf.summary.scalar(prefix + '_loss3', loss3))
    summaries.append(tf.summary.scalar(prefix + '_loss4', loss4))
    summaries.append(tf.summary.scalar(prefix + '_grad_loss1', grad_error1))
    summaries.append(tf.summary.scalar(prefix + '_grad_loss2', grad_error2))
    summaries.append(tf.summary.scalar(prefix + '_grad_loss3', grad_error3))
    summaries.append(tf.summary.scalar(prefix + '_grad_loss4', grad_error4))

    self.summ_op = tf.summary.merge(summaries)
    
def main(unused_argv):
  if FLAGS.output_dir == "":
    raise Exception("OUT_DIR must be specified")
  
  if os.path.exists(FLAGS.output_dir):
    raise Exception("OUT_DIR already exist")
    
  print 'Constructing models and inputs.'
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate)
    
    tower_grads = []
    itr_placeholders = []
    
    image1, image2, image3, file_name= build_tfrecord_input(training=True)
    
    split_image1 = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=image1)
    split_image2 = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=image2)
    split_image3 = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=image3)
    split_file_name = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=file_name)
        
    summaries_cpu = tf.get_collection(tf.GraphKeys.SUMMARIES, tf.get_variable_scope().name)

    with tf.variable_scope(tf.get_variable_scope()) as vs:
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          if i == FLAGS.num_gpus - 1:
            scopename = "model"
          else:
            scopename = '%s_%d' % ("tower", i)
          with tf.name_scope(scopename) as ns:
            if i == 0:
              model = Model(split_image1[i], split_image2[i], split_image3[i], split_file_name[i], reuse_scope=False, scope=vs)
            else:
              model = Model(split_image1[i], split_image2[i], split_image3[i], split_file_name[i], reuse_scope=True, scope=vs)
            
            loss = model.loss
            # Retain the summaries from the final tower.
            if i == FLAGS.num_gpus - 1:
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, ns)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = train_op.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            itr_placeholders.append(model.iter_num)
            
            
        # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = train_op.apply_gradients(grads)

    # Create a saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=5)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries + summaries_cpu)

    # Make training session.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    summary_writer = tf.summary.FileWriter(
        FLAGS.output_dir, graph=sess.graph, flush_secs=10)
  
    if FLAGS.pretrained_model:
      saver.restore(sess, FLAGS.pretrained_model)
      start_itr = int(FLAGS.pretrained_model.split("/")[-1][5:])
      sess.run(tf.local_variables_initializer())
    else:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      start_itr = 0
      
    tf.train.start_queue_runners(sess)
    
    # Run training.
    for itr in range(start_itr, FLAGS.num_iterations):
      # Generate new batch of data.
      feed_dict = {x:np.float32(itr) for x in itr_placeholders}
      _, summary_str = sess.run([apply_gradient_op, summary_op],
                                      feed_dict)
    
      if (itr) % SAVE_INTERVAL == 2:
        orig_image1, orig_image2, orig_image3, pred_flo, pred_warp, file_names = sess.run([model.orig_image1, 
                                                    model.orig_image2, 
                                                    model.orig_image3,
                                                    model.pred_flo,
                                                    model.pred_warp,
                                                    model.file_names],
                                                   feed_dict)
        
        #print(file_names)
        
        if (itr) % (SAVE_INTERVAL*10) == 2:
          tf.logging.info('Saving model.')
          saver.save(sess, FLAGS.output_dir + '/model' + str(itr))
        plot_flo_triple(orig_image1, orig_image2, orig_image3, pred_flo, pred_warp, file_names,
                 output_dir=FLAGS.output_dir, itr=itr)
        
#         eval_summary_str, eval_image, eval_mask_true, eval_mask_pred = sess.run([eval_model.summ_op, 
#                                                                                  eval_model.image, 
#                                                                                  eval_model.mask_true, 
#                                                                                  eval_model.mask_pred])
#         
#         plot_eval(eval_image, eval_mask_true, eval_mask_pred, 
#                   output_dir=FLAGS.output_dir, itr=itr)
        
      if (itr) % SUMMARY_INTERVAL:
        summary_writer.add_summary(summary_str, itr)
        
#       if (itr) % SUMMARY_INTERVAL*2 :
#         eval_summary_str = sess.run(eval_model.summ_op)
#         summary_writer.add_summary(eval_summary_str, itr)
        
    tf.logging.info('Saving model.')
    saver.save(sess, FLAGS.output_dir + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush
  

if __name__ == '__main__':
  app.run()
