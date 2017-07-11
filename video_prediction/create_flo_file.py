import os

# video_dirs = []
# data_dir = "/home/wangyang59/Data/ILSVRC2016_256/Data/VID/train/ILSVRC2015_VID_train_000"
# flo_dir = "/home/wangyang59/Data/videos_yi/flo/"
# # 
# # for i in range(4):
# #   data_diri = data_dir + str(i)
# #   tmp = os.listdir(data_diri)
# #   video_dirs += [os.path.join(data_diri, x) for x in tmp]
# #data_dir = "/home/yiyang/yi_videos/"
# data_dir = "/home/wangyang59/Data/videos_yi"
# tmp = os.listdir(data_dir)
# video_dirs = [os.path.join(data_dir, x) for x in tmp]
# 
image1 = []
image2 = []
flo = []
# 
# for video_dir in video_dirs:
#   images = sorted(os.listdir(video_dir))
#   dir_suffix = video_dir.split("/")[-1]
#   for i in range(len(images)-1):
#     image1.append(os.path.join(video_dir, images[i]))
#     image2.append(os.path.join(video_dir, images[i+1]))
#     flo.append(os.path.join(flo_dir, dir_suffix, images[i].split(".")[0] + ".flo"))
# 

data_dir = "/home/wangyang59/Data/FlyingChairs_release/data"
numbers = sorted(list(set([file.split("_")[0] for file in os.listdir(data_dir)])))

for number in numbers:
  image1.append(os.path.join(data_dir, number+"_img2.ppm"))
  image2.append(os.path.join(data_dir, number+"_img1.ppm"))
  flo.append(os.path.join(data_dir, number+"_flowr.flo"))

f = open('flownet2_file_list_chair.txt', 'w')
for i in range(len(image1)):
  f.write(image1[i] + " " + image2[i] + " " + flo[i] + '\n')  # python will convert \n to os.linesep
f.close()  # you can omit in most cases as the destructor will call it
