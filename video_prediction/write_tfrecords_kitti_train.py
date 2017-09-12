import os

import tensorflow as tf
import numpy as np

import multiprocessing
from PIL import Image
from random import shuffle, seed
import cv2
import StringIO
import png


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_flow_png(flow_file):
    """
    Read optical flow from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow

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
    image1_file, image2_file = file_name
    

    img=cv2.imread(image1_file)
    img2=cv2.imread(image2_file)
     
    img_all = np.concatenate((img, img2), axis=0)
    img_all_convert = hisEqulColor(img_all)
    img_convert, img2_convert = np.vsplit(img_all_convert, 2)
    
    cv2.imwrite("/home/wangyang59/convert1.jpeg", img_convert)
    cv2.imwrite("/home/wangyang59/convert2.jpeg", img2_convert)

    im = Image.open("/home/wangyang59/convert1.jpeg")
    im = im.crop((0, 0, 1224, 370))
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image1_raw = output.getvalue()
    output.close()
        

    im = Image.open("/home/wangyang59/convert2.jpeg")
    im = im.crop((0, 0, 1224, 370))
    output = StringIO.StringIO()
    im.save(output, format="jpeg")
    image2_raw = output.getvalue()
    output.close()
    
#     flo = read_flow_png(flo_file)
#     flo = flo[0:370, 0:1224, :]
    #occ = cv2.imread(occ_file, 0)
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image1_raw': _bytes_feature(image1_raw),
        'image2_raw': _bytes_feature(image2_raw),
#         'flo': _bytes_feature(flo.tostring()),
        #'occ': _bytes_feature(occ.tostring()),
        'file_name': _bytes_feature(image1_file)}))
    writer.write(example.SerializeToString())
  writer.close()  

def main(unused_argv):

  data_dir = "/home/wangyang59/Data/data_stereo_flow_multiview/"
  
  file_names = os.listdir(os.path.join(data_dir, "testing", "image_2"))
  scenes = sorted(set([file_name.split("_")[0] for file_name in file_names]))
  frames = sorted(set([file_name.split("_")[1].split(".")[0] for file_name in file_names]))
  print(scenes)
  print(frames)
  
  training_data = []
  
  for scene in scenes:
    for i in range(len(frames)-1):
      if frames[i] not in ["09", "10", "11", "12"]:
        for seg in ["training", "testing"]:
          for image in ["image_2", "image_3"]:
            file1 = os.path.join(data_dir, seg, image, scene+"_" + frames[i] + ".png")
            file2 = os.path.join(data_dir, seg, image, scene+"_"+ frames[i+1] + ".png")
            #flo_file = os.path.join(data_dir, "training", "flow_occ", scene+"_10"+".png")
            if os.path.exists(file1) and os.path.exists(file2):
              training_data.append([file1, file2])
          
  n = len(training_data)
  seed(42)
  shuffle(training_data)
    
  batch_size = 256
  inputs=[]
  
  for i in range(n/batch_size + 1):
    output_file = "/home/wangyang59/Data/ILSVRC2016_tf_kitti_2012_train_hist/%s.tfrecord" % i
    inputs.append((output_file, training_data[i*batch_size:(i+1)*batch_size]))
  
  for input in inputs:
    convert_to(input)

if __name__ == '__main__':
  tf.app.run()
