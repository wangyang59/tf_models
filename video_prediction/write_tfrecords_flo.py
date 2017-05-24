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
    image1_file, image2_file, flo_file = file_name
    im = resize_image(image1_file, 256)
    image1_raw = np.array(im.getdata(), dtype=np.float32).reshape((256, 256, 3)) / 255.0
    
    im = resize_image(image2_file, 256)
    image2_raw = np.array(im.getdata(), dtype=np.float32).reshape((256, 256, 3)) / 255.0
    
    flo_raw = crop_center(read_flow(flo_file), 256, 256)
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'image1_raw': _bytes_feature(image1_raw.tostring()),
        'image2_raw': _bytes_feature(image2_raw.tostring()),
        'flo_raw': _bytes_feature(flo_raw.tostring())}))
    writer.write(example.SerializeToString())
  writer.close()  

def main(unused_argv):
#   flo_root_dir = '/home/wangyang59/Data/ILSVRC2016_256_flo/'
#   flo_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(flo_root_dir) for f in filenames if os.path.splitext(f)[1] == '.flo']
#   shuffle(flo_files)
  with open("/home/wangyang59/Projects/flownet2/flownet2_file_list_shuf.txt") as f:
    file_names = f.readlines()
  
  file_names = [file_name.strip().split() for file_name in file_names]
  
  n = len(file_names)
  batch_size = 256
  inputs = []
  
  for i in range(n/batch_size):
    output_file = "/home/wangyang59/Data/ILSVRC2016_tf_flo/%s.tfrecord" % i
    flo_files_name = file_names[i*batch_size:(i+1)*batch_size]
    inputs.append((output_file, flo_files_name))
  
  pool = multiprocessing.Pool(20)
  pool.imap_unordered(convert_to, inputs, chunksize=10)
  pool.close()
  pool.join()

if __name__ == '__main__':
  tf.app.run()
