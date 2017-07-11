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
"""Code for training the prediction model."""

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from prediction_input_flo_chair import build_tfrecord_input, DATA_DIR
from prediction_model_flo_chair_ip import construct_model
from visualize import plot_flo_learn_symm
from optical_flow_warp import transformer
from optical_flow_warp_fwd import transformerFwd

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

flags.DEFINE_float('train_val_split', 0.95,
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

def mean_charb_error(true, pred, beta):
  return tf.reduce_sum(tf.sqrt((tf.square(beta*(true-pred)) + 0.001*0.001))) / tf.to_float(tf.size(pred))

def mean_charb_error_wmask(true, pred, mask, beta):
  return tf.reduce_sum(tf.sqrt((tf.square(beta*(true-pred)) + 0.001*0.001))*mask) / tf.to_float(tf.size(pred))


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

def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def cal_grad_error(flo, image, beta):
  """Calculate the gradient of the given image by calculate the difference between nearby pixels
  """
  error = 0.0
  img_grad_x = gradient_x(image)
  img_grad_y = gradient_y(image)
  
  weights_x = tf.exp(-10.0*tf.reduce_mean(tf.abs(img_grad_x), 3, keep_dims=True))
  weights_y = tf.exp(-10.0*tf.reduce_mean(tf.abs(img_grad_y), 3, keep_dims=True))
  
  error += mean_charb_error_wmask(flo[:, 1:, :, :], flo[:, :-1, :, :], weights_y, beta)
  error += mean_charb_error_wmask(flo[:, :, 1:, :], flo[:, :, :-1, :], weights_x, beta)
    
  return error / 2.0

def img_grad_error(true, pred, mask, beta):
  error = 0.0
  
  error += mean_charb_error_wmask(true[:, 1:, :, :] - true[:, :-1, :, :], 
                            pred[:, 1:, :, :] - pred[:, :-1, :, :], mask[:, 1:, :, :], beta)
  error += mean_charb_error_wmask(true[:, :, 1:, :] - true[:, :, :-1, :], 
                            pred[:, :, 1:, :] - pred[:, :, :-1, :], mask[:, :, 1:, :], beta)
  
  return error / 2.0
  

def epe(flo1, flo2):
  return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(flo1 - flo2), axis=3)))

def down_sample(image):
  batch_size, img_height, img_width, color_channels = map(int, image.get_shape()[0:4])
  kernel = np.array([1., 2., 1., 2., 4., 2., 1., 2., 1.], dtype=np.float32) / 16.0
  kernel = kernel.reshape((3, 3, 1, 1))
  kernel = tf.constant(kernel, shape=(3, 3, 1, 1), 
                       name='gaussian_kernel', verify_shape=True)
  
  blur_image = tf.nn.depthwise_conv2d(image, tf.tile(kernel, [1, 1, color_channels, 1]), 
                                           [1, 1, 1, 1], 'SAME')
  return tf.image.resize_bicubic(blur_image, [img_height/2, img_width/2])
  
def get_pyrimad(image):
  batch_size, img_height, img_width, color_channels = map(int, image.get_shape()[0:4])
  #image2 = down_sample(down_sample(image))
  #image3 = down_sample(image2)
  #image4 = down_sample(image3)
  #image5 = down_sample(image4)
  #image6 = down_sample(image5)
  
  image2 = tf.image.resize_area(image, [img_height/4, img_width/4])
  image3 = tf.image.resize_area(image, [img_height/8, img_width/8])
  image4 = tf.image.resize_area(image, [img_height/16, img_width/16])
  image5 = tf.image.resize_area(image, [img_height/32, img_width/32])
  image6 = tf.image.resize_area(image, [img_height/64, img_width/64])
  return image2, image3, image4, image5, image6
  
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


class Model_eval(object):

  def __init__(self,
               image1=None,
               image2=None,
               true_flo=None,
               scope=None,
               prefix="eval"):

    summaries = []
    
    batch_size, H, W, color_channels = map(int, image1.get_shape()[0:4])
    
    flow2, flow3, flow4, flow5, flow6 = construct_model(image1, image2)
        
    self.flo_loss = epe(true_flo, tf.image.resize_bicubic(20*flow2, [H, W]))


def main(unused_argv):
  if FLAGS.output_dir == "":
    raise Exception("OUT_DIR must be specified")
  
  if os.path.exists(FLAGS.output_dir):
    raise Exception("OUT_DIR already exist")
    
  print 'Constructing models and inputs.'
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
        
    eval_image1, eval_image2, eval_flo = build_tfrecord_input(training=False, num_epochs=1)
        
    with tf.variable_scope(tf.get_variable_scope()) as vs:
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          if i == FLAGS.num_gpus - 1:
            scopename = "model"
          else:
            scopename = '%s_%d' % ("tower", i)
          with tf.name_scope(scopename) as ns:
            
            # Retain the summaries from the final tower.
            if i == FLAGS.num_gpus - 1:
              eval_model = Model_eval(eval_image1, eval_image2, eval_flo, scope=vs)
            # Calculate the gradients for the batch of data on this CIFAR tower.
            
          

    # Create a saver.
    saver = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), max_to_keep=5)


    # Make training session.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    summary_writer = tf.summary.FileWriter(
        FLAGS.output_dir, graph=sess.graph, flush_secs=10)
  
    if FLAGS.pretrained_model:
      saver.restore(sess, FLAGS.pretrained_model)
      #start_itr = int(FLAGS.pretrained_model.split("/")[-1][5:])
      start_itr = 0
      sess.run(tf.local_variables_initializer())
    else:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.local_variables_initializer())
      start_itr = 0
      
    tf.train.start_queue_runners(sess)
    
    # Run training.
    flo_loss = 0.0
    for itr in range(start_itr, FLAGS.num_iterations):
      # Generate new batch of data.
      try:
        flo_loss += sess.run(eval_model.flo_loss)
        print(itr+1)
        print(flo_loss / (itr+1))
      except:
        print(itr+1)
        print(flo_loss / (itr+1))
        break
      
    tf.logging.flush
  

if __name__ == '__main__':
  app.run()
