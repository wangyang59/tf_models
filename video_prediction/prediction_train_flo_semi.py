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

from prediction_input_flo_res256 import build_tfrecord_input, DATA_DIR
from prediction_input_flo_eval import build_tfrecord_input_eval
from prediction_model_flo_res256 import construct_model
from visualize import plot_flo, plot_eval, plot_grad
from optical_flow_warp import transformer
import matplotlib

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
    error += mean_L1_error(cropped_image, shifted_image[:, 1:(img_height-1), 1:(img_width-1), :])
    
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

def threshold(t):
  return tf.where(t<0.5, x=tf.constant(0.0, shape=t.get_shape()), y=tf.constant(1.0, shape=t.get_shape()))

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
               flo=None,
               image=None,
               mask=None,
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
    
    
    if not reuse_scope:
      poss_move_mask1, bg_mask1 = construct_model(image1)
    else:  # If it's a validation or test model.
      with tf.variable_scope(scope, reuse=True):
        poss_move_mask1, bg_mask1 = construct_model(image1)
    
    with tf.variable_scope(scope, reuse=True):
      poss_move_mask2, bg_mask2 = construct_model(image2)
      poss_move_mask, bg_mask = construct_model(image)
    
    #supervised loss
#     sup_error = mean_L1_error(poss_move_mask, mask)
    sup_loss = -tf.reduce_mean(mask*tf.log(poss_move_mask+1e-10) + (1.0-mask)*tf.log(bg_mask+1e-10))
    
    batch_size, img_height, img_width = map(int, image1.get_shape()[0:3])
    
    # L2 loss, PSNR for eval.
    flo_grad_loss = cal_grad_error_wmask(flo, bg_mask1, self.kernels)
#     img_grad_loss = cal_grad_error_wmask(image1, poss_move_mask1, self.kernels)
#     img_grad_loss_we = cal_weighted_edge_diff(image1, poss_move_mask1, self.kernels, 1.0)
    
    #loss_mask_grad = cal_grad_error(1.0-poss_move_mask, self.kernels) * 1e-5
    var_loss = cal_weighted_var(flo, bg_mask1)
    #seg_loss = tf.reduce_sum(poss_move_mask) / tf.to_float(tf.size(poss_move_mask)) * 3.0
    #seg_loss = tf.reduce_sum(tf.square(poss_move_mask1)) / tf.to_float(tf.size(poss_move_mask1))
    seg_loss = tf.reduce_sum(tf.square(tf.reduce_sum(poss_move_mask1, axis=[1,2], keep_dims=True))) / tf.to_float(batch_size*img_height*img_height*img_width*img_width)

    poss_move_maskt = transformer(poss_move_mask2, flo, [img_height, img_width])
    move_mask_loss = mean_squared_error(poss_move_maskt, poss_move_mask1)
    
    summaries.append(tf.summary.scalar(prefix + '_loss_var', var_loss))
    summaries.append(tf.summary.scalar(prefix + '_loss_seg', seg_loss))
    summaries.append(tf.summary.scalar(prefix + '_loss_flo_grad', flo_grad_loss))
#     summaries.append(tf.summary.scalar(prefix + '_loss_img_grad', img_grad_loss))
    summaries.append(tf.summary.scalar(prefix + '_loss_moveMask', move_mask_loss))
#     summaries.append(tf.summary.scalar(prefix + '_sup_L1_error', sup_error))
    summaries.append(tf.summary.scalar(prefix + '_loss_sup', sup_loss))
    
    #self.loss = flo_grad_loss*0.0 + img_grad_loss*0.0 + move_mask_loss*0.0 + seg_loss*5.0+ var_loss*1.0 + sup_loss*1.0
    self.loss = seg_loss*10.0+ var_loss*1.0 + sup_loss*0.0 + flo_grad_loss*1.0
    self.orig_image1 = image1
    self.orig_image2 = image2
    self.flo = flo
    self.poss_move_mask1 = poss_move_mask1
    self.poss_move_mask2 = poss_move_mask2
    self.poss_move_maskt = poss_move_maskt
        
    self.flo_grad_bg_mask = tf.gradients(flo_grad_loss, [bg_mask1])[0]
#     self.img_grad_poss_move_mask = tf.gradients(img_grad_loss, [poss_move_mask1])[0]
#     self.img_grad_we_poss_move_mask = tf.gradients(img_grad_loss_we, [poss_move_mask1])[0]
    self.var_loss_bg_mask = tf.gradients(var_loss, [bg_mask1])[0]
#     self.sup_loss_poss_move_mask = tf.gradients(sup_error, [poss_move_mask])[0]
    self.move_loss_poss_move_mask = tf.gradients(move_mask_loss, [poss_move_mask1])[0]
    self.seg_loss_poss_move_mask = tf.gradients(seg_loss, [poss_move_mask1])[0]
    
    summaries.append(tf.summary.scalar(prefix + '_loss', self.loss))
    
    self.summ_op = tf.summary.merge(summaries)
    
class Model_eval(object):

  def __init__(self,
               image=None,
               mask=None,
               scope=None):

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
      
    batch_size, img_height, img_width = map(int, image.get_shape()[0:3])
    
    with tf.variable_scope(scope, reuse=True):
      poss_move_mask, bg_mask = construct_model(image)
    
    eval_error = mean_L1_error(threshold(poss_move_mask), mask)
    eval_loss = -tf.reduce_mean(mask*tf.log(poss_move_mask) + (1.0-mask)*tf.log(bg_mask))
    
#     true_img_grad_loss = cal_grad_error_wmask(image, mask, self.kernels)
#     pred_img_grad_loss = cal_grad_error_wmask(image, poss_move_mask, self.kernels)
    
    true_seg_loss = tf.reduce_sum(tf.square(tf.reduce_sum(mask, axis=[1,2], keep_dims=True))) / tf.to_float(batch_size*img_height*img_height*img_width*img_width)
    pred_seg_loss = tf.reduce_sum(tf.square(tf.reduce_sum(poss_move_mask, axis=[1,2], keep_dims=True))) / tf.to_float(batch_size*img_height*img_height*img_width*img_width)
    
    summaries.append(tf.summary.scalar('eval_error', eval_error))
    summaries.append(tf.summary.scalar('eval_loss', eval_loss))
#     summaries.append(tf.summary.scalar('eval_true_img_grad_loss', true_img_grad_loss))
#     summaries.append(tf.summary.scalar('eval_pred_img_grad_loss', pred_img_grad_loss))
    summaries.append(tf.summary.scalar('eval_true_seg_loss', true_seg_loss))
    summaries.append(tf.summary.scalar('eval_pred_seg_loss', pred_seg_loss))
    
    self.summ_op = tf.summary.merge(summaries)
    self.image = image
    self.mask_true = mask
    self.mask_pred = threshold(poss_move_mask)
    
def main(unused_argv):
  matplotlib.use('Agg')
  if FLAGS.output_dir == "":
    raise Exception("OUT_DIR must be specified")
  
  if os.path.exists(FLAGS.output_dir):
    raise Exception("OUT_DIR already exist")
    
  print 'Constructing models and inputs.'
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate)
    
    tower_grads = []
    itr_placeholders = []
    
    image, image2, flo= build_tfrecord_input(training=True)
    
    split_image = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=image)
    split_image2 = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=image2)
    split_flo = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=flo)
    
    image_sup, mask_sup = build_tfrecord_input_eval(training=True)
    split_image_sup = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=image_sup)
    split_mask_sup = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=mask_sup)
    
    image_eval, mask_eval = build_tfrecord_input_eval(training=False)
    
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
              model = Model(split_image[i], split_image2[i], split_flo[i], split_image_sup[i], split_mask_sup[i], 
                            reuse_scope=False, scope=vs)
            else:
              model = Model(split_image[i], split_image2[i], split_flo[i], split_image_sup[i], split_mask_sup[i],
                            reuse_scope=True, scope=vs)
            
            loss = model.loss
            # Retain the summaries from the final tower.
            if i == FLAGS.num_gpus - 1:
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, ns)
              eval_model = Model_eval(image_eval, mask_eval, scope=vs)

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
    else:
      sess.run(tf.global_variables_initializer())
      start_itr = 0
      
    tf.train.start_queue_runners(sess)
    
    # Run training.
    for itr in range(start_itr, FLAGS.num_iterations):
      # Generate new batch of data.
      feed_dict = {x:np.float32(itr) for x in itr_placeholders}
      _, summary_str = sess.run([apply_gradient_op, summary_op],
                                      feed_dict)
    
      if (itr) % SAVE_INTERVAL == 2:
        orig_image1, orig_image2, flo, poss_move_mask1, poss_move_mask2, poss_move_maskt, flo_grad_bg_mask, \
        var_loss_bg_mask, move_poss_move_mask, seg_loss_poss_move_mask = sess.run([model.orig_image1, 
                                                    model.orig_image2,
                                                    model.flo, 
                                                    model.poss_move_mask1,
                                                    model.poss_move_mask2,
                                                    model.poss_move_maskt,
                                                    model.flo_grad_bg_mask,
                                                    #model.img_grad_poss_move_mask,
                                                    #model.img_grad_we_poss_move_mask,
                                                    model.var_loss_bg_mask,
                                                    #model.sup_loss_poss_move_mask,
                                                    model.move_loss_poss_move_mask,
                                                    model.seg_loss_poss_move_mask],
                                                    feed_dict)
        if (itr) % SAVE_INTERVAL*10 == 2:  
          tf.logging.info('Saving model.')
          saver.save(sess, FLAGS.output_dir + '/model' + str(itr))
        plot_flo(orig_image1, orig_image2, flo, poss_move_mask1, poss_move_mask2, poss_move_maskt,
                 output_dir=FLAGS.output_dir, itr=itr)
        plot_grad(var_loss_bg_mask, seg_loss_poss_move_mask, move_poss_move_mask, flo_grad_bg_mask,
                  output_dir=FLAGS.output_dir, itr=itr)
        
        eval_summary_str, eval_image, eval_mask_true, eval_mask_pred = sess.run([eval_model.summ_op, 
                                                                                 eval_model.image, 
                                                                                 eval_model.mask_true, 
                                                                                 eval_model.mask_pred])
        
        plot_eval(eval_image, eval_mask_true, eval_mask_pred, 
                  output_dir=FLAGS.output_dir, itr=itr)
        
      if (itr) % SUMMARY_INTERVAL:
        summary_writer.add_summary(summary_str, itr)
        
      if (itr) % SUMMARY_INTERVAL*5 :
        eval_summary_str = sess.run(eval_model.summ_op)
        summary_writer.add_summary(eval_summary_str, itr)
        
    tf.logging.info('Saving model.')
    saver.save(sess, FLAGS.output_dir + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush
  

if __name__ == '__main__':
  app.run()
