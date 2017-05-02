import os
import multiprocessing
import functools

def stablize(out_dir, vide_file):
  name = vide_file.split("/")[-1][:-4]
  out = os.path.join(out_dir, name)
  os.system( '/home/wangyang59/opencv-3.2.0/samples/cpp/example_cmake/build/opencv_example %s --out-dir=%s' % (vide_file,  out))

def main():
  data_dir = '/home/wangyang59/Data/ILSVRC2016/Data/VID/snippets/train/ILSVRC2015_VID_train_0001/'
  video_files = os.listdir(data_dir)
  
#   for h5_file in h5_files[0:10]:
#     convert_to(os.path.join(h5_dir, h5_file), "/home/wangyang59/Data/ILSVRC2016_tf/train")
  
  
  
  fun = functools.partial(stablize, "/home/wangyang59/Data/ILSVRC2016_stab/train/")
  pool = multiprocessing.Pool(20)
  pool.imap_unordered(fun, [os.path.join(data_dir, video_file) for video_file in video_files], chunksize=10)
  pool.close()
  pool.join()
  
if __name__ == '__main__':
  main()
