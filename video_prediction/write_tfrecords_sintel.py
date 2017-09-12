import os

import tensorflow as tf
import numpy as np

import multiprocessing
from PIL import Image
from random import shuffle, seed
import cv2
import StringIO
import time


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



def hisEqulColor(img):
  ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
  channels=cv2.split(ycrcb)
  cv2.equalizeHist(channels[0],channels[0])
  cv2.merge(channels,ycrcb)
  cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
  return img


def convert_to(input_tuple):
  out_name, file_names = input_tuple
  writer = tf.python_io.TFRecordWriter(out_name)
  print('Writing', out_name)
  
  for file_name in file_names:
    image1_file, image2_file, flo_file, occ_file, scene, file_no = file_name
    
#     img=cv2.imread(image1_file)
#     img2=cv2.imread(image2_file)
#      
#     img_all = np.concatenate((img, img2), axis=0)
#     start = time.time()
#     img_all_convert = hisEqulColor(img_all)
#     print(time.time() - start)
#     img_convert, img2_convert = np.vsplit(img_all_convert, 2)
#     
#     cv2.imwrite("/home/wangyang59/convert1.jpeg", img_convert)
#     cv2.imwrite("/home/wangyang59/convert2.jpeg", img2_convert)

    #im = Image.open("/home/wangyang59/convert1.jpeg")
    im = Image.open(image1_file)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image1_raw = output.getvalue()
    output.close()
        
    #im = Image.open("/home/wangyang59/convert2.jpeg")
    im = Image.open(image2_file)
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image2_raw = output.getvalue()
    output.close()
    
    flo = read_flow(flo_file)
    occ = cv2.imread(occ_file, 0)
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image1_raw': _bytes_feature(image1_raw),
        'image2_raw': _bytes_feature(image2_raw),
        'flo': _bytes_feature(flo.tostring()),
        'occ': _bytes_feature(occ.tostring()),
        'scene': _bytes_feature(scene),
        'file_no': _bytes_feature(file_no)}))
    writer.write(example.SerializeToString())
  writer.close()  

def main(unused_argv):
  data_dir = "/home/wangyang59/Data/mpi-sintel/training"
  
  scenes = os.listdir(os.path.join(data_dir, "final"))
  training_data = []
  
  for scene in scenes:
    image_files = sorted(os.listdir(os.path.join(data_dir, "final", scene)))
    for i in range(len(image_files) - 1):
      image1 = image_files[i].split(".")[0]
      image2 = image_files[i+1].split(".")[0]
      training_data.append([os.path.join(data_dir, "clean", scene, image1+".png"), 
                     os.path.join(data_dir, "clean", scene, image2+".png"),
                     os.path.join(data_dir, "flow", scene, image1+".flo"),
                     os.path.join(data_dir, "occlusions", scene, image1+".png"),
                     scene,
                     image1])
    
  n = len(training_data)
  seed(42)
  shuffle(training_data)
  batch_size = 32
  inputs=[]
  
  for i in range(n/batch_size + 1):
    output_file = "/home/wangyang59/Data/ILSVRC2016_tf_sintel_clean_train/%s.tfrecord" % i
    inputs.append((output_file, training_data[i*batch_size:(i+1)*batch_size]))
  
  for input in inputs:
    convert_to(input)

if __name__ == '__main__':
  tf.app.run()
