import os

import tensorflow as tf
import numpy as np

import multiprocessing
from PIL import Image
from random import shuffle

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

def convert_to(out_name, data_files):
  writer = tf.python_io.TFRecordWriter(out_name)
  print('Writing', out_name)
  
  for data_file in data_files:
    image_file, gt_file = data_file
    
    im = resize_image(image_file, 256)
    image_raw = np.array(im.getdata(), dtype=np.float32).reshape((256, 256, 3)) / 255.0
    
    im = resize_image(gt_file, 256)
    mask_raw = np.array(im.getdata(), dtype=np.float32).reshape((256, 256, 1)) / 255.0
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw.tostring()),
        'mask_raw': _bytes_feature(mask_raw.tostring())}))
    writer.write(example.SerializeToString())
  writer.close() 

def main(unused_argv):
  img_dir = '/home/wangyang59/Data/saliency_data/HKU-IS/imgs'
  gt_dir = '/home/wangyang59/Data/saliency_data/HKU-IS/gt'
  file_names = sorted(os.listdir(img_dir))
  
  data_files = [(os.path.join(img_dir, file_name), os.path.join(gt_dir, file_name)) for file_name in file_names]
  
  out_file = "/home/wangyang59/Data/ILSVRC2016_tf_eval/hku1.tfrecord"
  convert_to(out_file, data_files[0:256])

if __name__ == '__main__':
  tf.app.run()
