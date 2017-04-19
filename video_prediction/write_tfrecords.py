import os
import h5py

import tensorflow as tf
import numpy as np

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(h5_file, out_dir):
  """Converts a dataset to tfrecords."""
  sequence_length = 10
  f = h5py.File(h5_file, "r")
  images = f["data"].value
  num_frames = images.shape[0]
  name = h5_file.split("/")[-1][:-3]
  
  if num_frames < sequence_length:
    return
  
  filename = os.path.join(out_dir, name + ".tfrecord")
  writer = tf.python_io.TFRecordWriter(filename)
  print('Writing', filename)

  for i in range(num_frames / sequence_length):
    image_raw = np.zeros((sequence_length, 256, 256, 3), dtype = np.float32)
    for j in range(sequence_length):
      image_raw[j, :, :, :] = images[i*sequence_length+j].reshape((256, 256, 3), order='F').transpose(1, 0, 2,) 
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw.tostring())}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  h5_dir = '/home/wangyang59/Data/ILSVRC2016_h5/train/'
  h5_files = os.listdir(h5_dir)
  
  for h5_file in h5_files[0:10]:
    convert_to(os.path.join(h5_dir, h5_file), "/home/wangyang59/Data/ILSVRC2016_tf/train")

if __name__ == '__main__':
  tf.app.run()
