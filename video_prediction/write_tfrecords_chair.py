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

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print 'Magic number incorrect. Invalid .flo file'
        raise ValueError
    else:
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        #print "Reading %d x %d flo file" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h, w, 2))
    f.close()
    return data2d

def convert_to(input_tuple):
  out_name, file_names = input_tuple
  writer = tf.python_io.TFRecordWriter(out_name)
  print('Writing', out_name)
  
  for file_name in file_names:
    image1_file = file_name + "_img1.jpeg"
    image2_file = file_name + "_img2.jpeg"
    flo_file = file_name + "_flow.flo"
    
    im = Image.open(image1_file)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image1_raw = output.getvalue()
    output.close()
        
    im = Image.open(image2_file)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image2_raw = output.getvalue()
    output.close()
    
    flo = read_flow(flo_file)
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image1_raw': _bytes_feature(image1_raw),
        'image2_raw': _bytes_feature(image2_raw),
        'flo': _bytes_feature(flo.tostring())}))
    writer.write(example.SerializeToString())
  writer.close()  

def main(unused_argv):
  data_dir = "/home/wangyang59/Data/FlyingChairs_release/data"
  
  image_files = sorted(set([file.split("_")[0] for file in os.listdir(data_dir)]))
  image_files = [os.path.join(data_dir, image_file) for image_file in image_files]
  
  n = len(image_files)
  print(n)
  batch_size = 512
  inputs = []
  
  for i in range(n/batch_size):
    output_file = "/home/wangyang59/Data/ILSVRC2016_tf_chair_hist/%s.tfrecord" % i
    inputs.append((output_file, image_files[i*batch_size:(i+1)*batch_size]))
    
  pool = multiprocessing.Pool(20)
  pool.imap_unordered(convert_to, inputs, chunksize=1)
  pool.close()
  pool.join()

if __name__ == '__main__':
  tf.app.run()
