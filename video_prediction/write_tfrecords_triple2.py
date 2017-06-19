import os

import tensorflow as tf
import numpy as np

import multiprocessing
from PIL import Image
from random import shuffle

import StringIO


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

def crop_center(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx, :]

def convert_to(input_tuple):
  out_name, file_names = input_tuple
  writer = tf.python_io.TFRecordWriter(out_name)
  print('Writing', out_name)
  
  for file_name in file_names:
    image1_file, image2_file, image3_file = file_name
    
    im = resize_image(image1_file, 256)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image1_raw = output.getvalue()
    output.close()
        
    im = resize_image(image2_file, 256)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image2_raw = output.getvalue()
    output.close()
    
    im = resize_image(image3_file, 256)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image3_raw = output.getvalue()
    output.close()
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image1_raw': _bytes_feature(image1_raw),
        'image2_raw': _bytes_feature(image2_raw),
        'image3_raw': _bytes_feature(image3_raw),
        'file_name': _bytes_feature(image1_file + "," + image2_file)}))
    writer.write(example.SerializeToString())
  writer.close()  

def main(unused_argv):
  video_dirs = []
  data_dir = "/home/wangyang59/Data/ILSVRC2016_256/Data/VID/train/ILSVRC2015_VID_train_000"
  
  for i in range(4):
    data_diri = data_dir + str(i)
    tmp = os.listdir(data_diri)
    video_dirs += [os.path.join(data_diri, x) for x in tmp]
  
  image_files = []
  
  for video_dir in video_dirs:
    images = sorted(os.listdir(video_dir))
    for i in range(len(images)/3):
      image_files.append((os.path.join(video_dir, images[i*3]), os.path.join(video_dir, images[i*3+1]), os.path.join(video_dir, images[i*3+2])))
    
  shuffle(image_files)
  
  n = len(image_files)
  print(n)
  batch_size = 256*10
  inputs = []
  
  for i in range(n/batch_size):
    output_file = "/home/wangyang59/Data/ILSVRC2016_tf_triple/%s.tfrecord" % i
    inputs.append((output_file, image_files[i*batch_size:(i+1)*batch_size]))
    
  pool = multiprocessing.Pool(20)
  pool.imap_unordered(convert_to, inputs, chunksize=2)
  pool.close()
  pool.join()

if __name__ == '__main__':
  tf.app.run()
