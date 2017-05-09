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

from prediction_input import build_tfrecord_input
from prediction_model import construct_model
from spatial_transformer import transformer
from visualize import plot_gif

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

# tf record data location:
DATA_DIR = '/home/wangyang59/Data/robot/push/push_train'

# local output directory
OUT_DIR = '/home/wangyang59/Projects/tf_models/video_prediction/tmp/data/stp_run'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', OUT_DIR, 'directory for model checkpoints.')
flags.DEFINE_string('event_log_dir', OUT_DIR, 'directory for writing summary.')
flags.DEFINE_integer('num_iterations', 100000, 'number of training iterations.')
flags.DEFINE_string('pretrained_model', '',
                    'filepath of a pretrained model to initialize from.')

flags.DEFINE_integer('sequence_length', 10,
                     'sequence length, including context frames.')
flags.DEFINE_integer('context_frames', 2, '# of frames before predictions.')
flags.DEFINE_integer('use_state', 1,
                     'Whether or not to give the state+action to the model')

flags.DEFINE_string('model', 'CDNA',
                    'model architecture to use - CDNA, DNA, or STP')

flags.DEFINE_integer('num_masks', 10,
                     'number of masks, usually 1 for DNA, 10 for CDNA, STN.')
flags.DEFINE_float('schedsamp_k', 900.0,
                   'The k hyperparameter for scheduled sampling,'
                   '-1 for no scheduled sampling.')
flags.DEFINE_float('train_val_split', 0.95,
                   'The percentage of files to use for the training set,'
                   ' vs. the validation set.')

flags.DEFINE_integer('batch_size', 32, 'batch size for training')
flags.DEFINE_float('learning_rate', 0.001,
                   'the base learning rate of the generator')
flags.DEFINE_integer('num_gpus', 1,
                   'the number of gpu to use')

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


class Model(object):

  def __init__(self,
               images=None,
               actions=None,
               states=None,
               sequence_length=None,
               reuse_scope=None,
               prefix="train"):

    if sequence_length is None:
      sequence_length = FLAGS.sequence_length

    self.iter_num = tf.placeholder(tf.float32, [])

    # Split into timesteps.
    actions = tf.split(axis=1, num_or_size_splits=int(actions.get_shape()[1]), value=actions)
    actions = [tf.squeeze(act) for act in actions]
    states = tf.split(axis=1, num_or_size_splits=int(states.get_shape()[1]), value=states)
    states = [tf.squeeze(st) for st in states]
    images = tf.split(axis=1, num_or_size_splits=int(images.get_shape()[1]), value=images)
    images = [tf.squeeze(img) for img in images]
    
    stp_kernel = tf.convert_to_tensor(
      np.tile(np.array([0.9, -0.1, 0.0, 
                        0.1, 0.9, 0.0], np.float32), [32, 1]))
    
    gen_images = [transformer(image, stp_kernel, (64, 64)) for image in images]
    self.gen_images = gen_images
    self.orig_images = images


def main(unused_argv):

  print 'Constructing models and inputs.'
  with tf.variable_scope('model', reuse=None) as training_scope:
    images, actions, states = build_tfrecord_input(training=True)
    model = Model(images, actions, states, FLAGS.sequence_length)

  print 'Constructing saver.'
  # Make saver.

  # Make training session.
  sess = tf.InteractiveSession()

  tf.train.start_queue_runners(sess)
  sess.run(tf.global_variables_initializer())

  tf.logging.info('iteration number, cost')

  # Run training.
  for itr in range(FLAGS.num_iterations):
    # Generate new batch of data.
    feed_dict = {model.iter_num: np.float32(itr)}

    if (itr) % SAVE_INTERVAL == 2:
      tf.logging.info('Saving model.')
      orig_images, gen_images = sess.run([model.orig_images, model.gen_images], feed_dict)
      mask_lists = [[np.zeros((32, 64, 64, 1)) for jj in range(26)] for ii in range(10)]
      shifted_masks = [np.zeros((32, 64, 64, 1)) for ii in range(10)]
      poss_move_masks = [np.zeros((32, 64, 64, 1)) for ii in range(10)]
      plot_gif(orig_images, gen_images, shifted_masks, mask_lists, poss_move_masks,
               output_dir=FLAGS.output_dir, itr=itr)
      
      return




if __name__ == '__main__':
  app.run()
