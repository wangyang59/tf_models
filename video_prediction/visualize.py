import moviepy.editor as mpy
import os
import cPickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def merge(masks, orig_image, gen_image, shifted_mask, batch_num, gen_image_max):
  grey_cmap = plt.get_cmap("Greys")
  seis_cmap = plt.get_cmap("seismic")
  
  assert len(masks) == 26
  figures = masks[2:14] + [masks[0]] + masks[14:26] + [masks[1]] + [shifted_mask, orig_image, gen_image]
  h = 6
  w = 5
  img_size = 64
  gap = 3
  img = np.zeros((h * (img_size + gap), w * (img_size + gap), 3))
  for idx in xrange(len(figures)):
    i = idx % w
    j = idx // w
    if idx < len(masks):
      tmp = grey_cmap(figures[idx][batch_num][:, :, 0])[:, :, 0:3]
    elif idx == len(masks):
      tmp = seis_cmap(figures[idx][batch_num])[:, :, 0, 0:3]
    elif idx == len(masks) + 1:
      tmp = figures[idx][batch_num]
    else:
      tmp = figures[idx][batch_num] / gen_image_max
    img[j*(img_size+gap):j*(img_size+gap)+img_size, i*(img_size+gap):i*(img_size+gap)+img_size, :] = \
        tmp * 255.0
  
  return img

def plot_gif(orig_images, gen_images, shifted_masks, mask_lists, output_dir, itr):
  assert len(orig_images) == len(gen_images)
  assert len(orig_images) == len(shifted_masks)
  assert len(orig_images) == len(mask_lists)
  
  batch_size = orig_images[0].shape[0]
  os.mkdir(os.path.join(output_dir, "itr_" + str(itr)))
  
  gen_image_max = np.max([np.max(x) for x in gen_images])
  shifted_masks = [x - 0.5 for x in shifted_masks]
  #cmap = plt.get_cmap('seismic')
  
  for i in range(batch_size):
    video = []
    for j in range(len(orig_images)):
      video.append(merge(mask_lists[j], orig_images[j], 
                         gen_images[j], shifted_masks[j], i, gen_image_max))
    clip = mpy.ImageSequenceClip(video, fps=2)
    clip.write_gif(os.path.join(output_dir, "itr_"+str(itr), "All_batch_" + str(i) + ".gif"),
                   verbose=False)
#     clip = mpy.ImageSequenceClip([x[i]*255.0 for x in orig_images], fps=5)
#     clip.write_gif(os.path.join(output_dir, "itr_"+str(itr), "orig_images_batch_" + str(i) + ".gif"),
#                    verbose=False)
#     
#     clip = mpy.ImageSequenceClip([x[i]/gen_image_max*255.0 for x in gen_images], fps=5)
#     clip.write_gif(os.path.join(output_dir, "itr_"+str(itr), "gen_images_batch_" + str(i)) + ".gif",
#                    verbose=False)
#     
#     clip = mpy.ImageSequenceClip([(cmap(x[i]))[:, :, 0, 0:3]*255.0 for x in shifted_masks], fps=5)
#     clip.write_gif(os.path.join(output_dir, "itr_"+str(itr), "masks_batch_" + str(i)) + ".gif",
#                    verbose=False)
    
  with open(os.path.join(output_dir, "itr_"+str(itr), "shifted_mask.pickle"), "wb") as f:
    cPickle.dump(shifted_masks, f)
  
#   with open(os.path.join(output_dir, "itr_"+str(itr), "mask_lists.pickle"), "wb") as f:
#     cPickle.dump(mask_lists, f)
    
def npy_to_gif(npy, filename):
    clip = mpy.ImageSequenceClip([x*255.0 for x in npy], fps=10)
    clip.write_gif(filename)
    
def main():
#   video = []
#   for i in range(8):
#     image = np.ones(shape=[64, 64, 1], dtype=np.float32)
#     image = image * 1.0
#     image[i*8, i*8, 0] = 0.8
#     video.append(image)
#   clip = mpy.ImageSequenceClip([x * 255.0 for x in video], fps=5)
#   clip.write_gif("./test.gif", verbose=False)
  def merge(masks, batch_num, cmap):
    assert len(masks) == 26
    masks = masks[2:14] + [masks[0]] + masks[14:26] + [masks[1]]
    h = 6
    w = 5
    img_size = 64
    gap = 3
    img = np.zeros((h * (img_size + gap), w * (img_size + gap), 3))
    for idx in xrange(len(masks)):
      i = idx % w
      j = idx // w
      img[j*(img_size+gap):j*(img_size+gap)+img_size, i*(img_size+gap):i*(img_size+gap)+img_size, :] = \
          cmap(masks[idx][batch_num][:, :, 0])[:, :, 0:3] * 255.0
    return img#.astype('uint8')

  with open("./tmp/data/my_multigpu_run_k=0_test/itr_40002/mask_lists.pickle") as f:
    mask_lists = cPickle.load(f)
  
  video = []
  cmap = plt.get_cmap('Greys')
  
  for i in range(8):
    video.append(merge(mask_lists[i], 1, cmap))
  clip = mpy.ImageSequenceClip(video, fps=1)
  clip.write_gif("./test.gif", verbose=False)
  
  
if __name__ == '__main__':
  main()