import os
import multiprocessing
import cv2
import numpy as np

def hisEqulColor(img):
  ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
  channels=cv2.split(ycrcb)
  cv2.equalizeHist(channels[0],channels[0])
  cv2.merge(channels,ycrcb)
  cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
  return img


def convert_to(file_name):
  image1_file = file_name + "_img1.ppm"
  image2_file = file_name + "_img2.ppm"
  
  img=cv2.imread(image1_file)
  img2=cv2.imread(image2_file)
  
  img_all = np.concatenate((img, img2), axis=0)
  
  img_all_convert = hisEqulColor(img_all)
  
  img_convert, img2_convert = np.vsplit(img_all_convert, 2)
  
  cv2.imwrite(file_name + "_img1.jpeg", img_convert)
  cv2.imwrite(file_name + "_img2.jpeg", img2_convert)
  
  print(file_name)


def main():
  data_dir = "/home/wangyang59/Data/FlyingChairs_release/data"
  
  image_files = sorted(set([file.split("_")[0] for file in os.listdir(data_dir)]))
  image_files = [os.path.join(data_dir, image_file) for image_file in image_files]
  
  n = len(image_files)
  print(n)
    
  pool = multiprocessing.Pool(20)
  pool.imap_unordered(convert_to, image_files, chunksize=20)
  pool.close()
  pool.join()

if __name__ == '__main__':
  main()


