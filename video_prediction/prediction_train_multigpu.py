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

from prediction_input2 import build_tfrecord_input
from prediction_model2 import construct_model
from visualize import plot_gif

import os

# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 40

# How often to run a batch through the validation model.
VAL_INTERVAL = 200

# How often to save a model checkpoint
SAVE_INTERVAL = 2000

# tf record data location:
#DATA_DIR = '/home/wangyang59/Data/robot/push/push_train'
DATA_DIR = '/home/wangyang59/Data/ILSVRC2016_tf/train'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('output_dir', "", 'directory for model checkpoints.')
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

flags.DEFINE_integer('num_masks', 26,
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
               images=None,
               actions=None,
               states=None,
               sequence_length=None,
               reuse_scope=None,
               prefix="train"):

    if sequence_length is None:
      sequence_length = FLAGS.sequence_length

    #self.prefix = prefix = tf.placeholder(tf.string, [])
    self.iter_num = tf.placeholder(tf.float32, [])
    summaries = []

    # Split into timesteps.
    actions = tf.split(axis=1, num_or_size_splits=int(actions.get_shape()[1]), value=actions)
    actions = [tf.squeeze(act) for act in actions]
    states = tf.split(axis=1, num_or_size_splits=int(states.get_shape()[1]), value=states)
    states = [tf.squeeze(st) for st in states]
    images = tf.split(axis=1, num_or_size_splits=int(images.get_shape()[1]), value=images)
    images = [tf.squeeze(img) for img in images]

    if reuse_scope is None:
      gen_images, gen_states, shifted_masks, mask_lists, entropy_losses = construct_model(
          images,
          actions,
          states,
          iter_num=self.iter_num,
          k=FLAGS.schedsamp_k,
          use_state=FLAGS.use_state,
          num_masks=FLAGS.num_masks,
          cdna=FLAGS.model == 'CDNA',
          dna=FLAGS.model == 'DNA',
          stp=FLAGS.model == 'STP',
          context_frames=FLAGS.context_frames)
    else:  # If it's a validation or test model.
      with tf.variable_scope(reuse_scope, reuse=True):
        gen_images, gen_states, shifted_masks, mask_lists, entropy_losses = construct_model(
            images,
            actions,
            states,
            iter_num=self.iter_num,
            k=FLAGS.schedsamp_k,
            use_state=FLAGS.use_state,
            num_masks=FLAGS.num_masks,
            cdna=FLAGS.model == 'CDNA',
            dna=FLAGS.model == 'DNA',
            stp=FLAGS.model == 'STP',
            context_frames=FLAGS.context_frames)
    
    entropy_loss = tf.reduce_mean(entropy_losses[FLAGS.context_frames - 1:]) * 0.0#1e-2
    # L2 loss, PSNR for eval.
    loss, psnr_all = 0.0, 0.0
    for i, x, gx in zip(
        range(len(gen_images)), images[FLAGS.context_frames:],
        gen_images[FLAGS.context_frames - 1:]):
      recon_cost = mean_squared_error(x, gx)
      psnr_i = peak_signal_to_noise_ratio(x, gx)
      psnr_all += psnr_i
      summaries.append(
          tf.summary.scalar(prefix + '_recon_cost' + str(i), recon_cost))
      summaries.append(tf.summary.scalar(prefix + '_psnr' + str(i), psnr_i))
      #summaries.append(tf.summary.image("gen_image" + str(i), gx, max_outputs=1))
      #summaries.append(tf.summary.image("orig_image" + str(i), x, max_outputs=1))
      #self.orig_images.append(x[0])
      #self.gen_images.append(gx[0])
      loss += recon_cost
    
    self.orig_images = images[FLAGS.context_frames:]
    self.gen_images = gen_images[FLAGS.context_frames - 1:]
    self.shifted_masks = shifted_masks[FLAGS.context_frames - 1:]
    self.mask_lists = mask_lists[FLAGS.context_frames - 1:]
    
    for i, state, gen_state in zip(
        range(len(gen_states)), states[FLAGS.context_frames:],
        gen_states[FLAGS.context_frames - 1:]):
      state_cost = mean_squared_error(state, gen_state) * 1e-4
      summaries.append(
          tf.summary.scalar(prefix + '_state_cost' + str(i), state_cost))
      loss += state_cost
    summaries.append(tf.summary.scalar(prefix + '_psnr_all', psnr_all))
    self.psnr_all = psnr_all
    
    summaries.append(tf.summary.scalar(prefix + '_entropy_loss', entropy_loss))

    loss = loss / np.float32(len(images) - FLAGS.context_frames)

    summaries.append(tf.summary.scalar(prefix + '_loss', loss))
    
    self.loss = loss + entropy_loss
    summaries.append(tf.summary.scalar(prefix + '_loss_with_entropy', self.loss))
    
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
    
    images, actions, states = build_tfrecord_input(training=True)
    split_images = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
    split_actions = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=actions)
    split_states = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=states)
  
    with tf.variable_scope(tf.get_variable_scope()) as vs:
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          if i == FLAGS.num_gpus - 1:
            scopename = "model"
          else:
            scopename = '%s_%d' % ("tower", i)
          with tf.name_scope(scopename) as ns:
            if i == 0:
              model = Model(split_images[i], split_actions[i], split_states[i], FLAGS.sequence_length)
            else:
              model = Model(split_images[i], split_actions[i], split_states[i], FLAGS.sequence_length, vs)
            
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
    summary_op = tf.summary.merge(summaries)

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
        orig_images, gen_images, shifted_masks, mask_lists = sess.run([model.orig_images, 
                                                           model.gen_images, 
                                                           model.shifted_masks,
                                                           model.mask_lists],
                                                          feed_dict)
        tf.logging.info('Saving model.')
        saver.save(sess, FLAGS.output_dir + '/model' + str(itr))
        plot_gif(orig_images, gen_images, shifted_masks, mask_lists, 
                 output_dir=FLAGS.output_dir, itr=itr)
  
      if (itr) % SUMMARY_INTERVAL:
        summary_writer.add_summary(summary_str, itr)
  
    tf.logging.info('Saving model.')
    saver.save(sess, FLAGS.output_dir + '/model')
    tf.logging.info('Training complete')
    tf.logging.flush
  

if __name__ == '__main__':
  app.run()