import os

import tensorflow as tf
import numpy as np

import multiprocessing
import functools
from PIL import Image

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def resize_image(image_file, input_dir, output_dir):
def resize_image(image_file, size):
    im = Image.open(image_file)
    if im.size[0] < im.size[1]:
        new_height = size
        new_width = int(round(float(size) / im.size[0] * im.size[1]))
    else:
        new_width = size
        new_height = int(round(float(size) / im.size[1] * im.size[0]))
    im = im.resize((new_height, new_width), Image.ANTIALIAS)
    
    half_image_size = size / 2
    half_the_width = im.size[0] / 2
    half_the_height = im.size[1] / 2
    im = im.crop((half_the_width - half_image_size,
                  half_the_height - half_image_size,
                  half_the_width + half_image_size,
                  half_the_height + half_image_size))
    return im

def convert_to(out_dir, video_dir):
  """Converts a dataset to tfrecords."""
  sequence_length = 9
  name = video_dir.split("/")[-1]
  splits = os.listdir(video_dir)
  
  filename = os.path.join(out_dir, name + ".tfrecord")
  writer = tf.python_io.TFRecordWriter(filename)
  print('Writing', filename)
  
  for split in splits:
    images = sorted(os.listdir(os.path.join(video_dir, split)))
    assert len(images) == sequence_length
    image_raw = np.zeros((sequence_length, 256, 256, 3), dtype = np.float32)
    
    for cnt, image in enumerate(images):
      im = resize_image(os.path.join(video_dir, split, image), 256)
      image_raw[cnt] = np.array(im.getdata(), dtype=np.float32).reshape((256, 256, 3)) / 255.0
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw.tostring())}))
    writer.write(example.SerializeToString())
  writer.close()


def main(unused_argv):
  data_dir = '/home/wangyang59/Data/ILSVRC2016_stab/train/'
  video_dirs = os.listdir(data_dir)
  
#  convert_to("/home/wangyang59/Data/ILSVRC2016_tf_stab/train", os.path.join(data_dir, video_dirs[0]))  
    
  fun = functools.partial(convert_to, "/home/wangyang59/Data/ILSVRC2016_tf_stab/train")
  pool = multiprocessing.Pool(20)
  pool.imap_unordered(fun, [os.path.join(data_dir, video_dir) for video_dir in video_dirs], chunksize=10)
  pool.close()
  pool.join()

if __name__ == '__main__':
  tf.app.run()
