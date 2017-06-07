import moviepy.editor as mpy
import os
import cPickle
import numpy as np
import matplotlib.pyplot as plt
from flowlib import flow_to_image
from PIL import Image

def blow_up(kernel, size):
  kernel_h, kernel_w = kernel.shape
  img = np.zeros((size, size), dtype=np.float32)
  block_h = size / kernel_h
  block_w = size / kernel_w
  for i in range(kernel_h):
    for j in range(kernel_w):
      img[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w] = kernel[i, j]
  return img

def merge(masks, orig_image, gen_image, shifted_mask, poss_move_mask, batch_num, gen_image_max):
  grey_cmap = plt.get_cmap("Greys")
  seis_cmap = plt.get_cmap("seismic")
  
  assert len(masks) == 26
  figures = masks[2:14] + [masks[0]] + masks[14:26] + [masks[1]] + [shifted_mask, orig_image, gen_image, poss_move_mask]
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
    elif idx == len(masks) + 2:
      tmp = figures[idx][batch_num] / gen_image_max
    #else:
    #  tmp = grey_cmap(blow_up(figures[idx][batch_num], img_size))[:, :, 0:3]
    else:
      tmp = grey_cmap(figures[idx][batch_num][:, :, 0])[:, :, 0:3]
    img[j*(img_size+gap):j*(img_size+gap)+img_size, i*(img_size+gap):i*(img_size+gap)+img_size, :] = \
        tmp * 255.0
  
  return img

def plot_gif(orig_images, gen_images, shifted_masks, mask_lists, poss_move_masks, output_dir, itr):
  assert len(orig_images) == len(gen_images)
  assert len(orig_images) == len(shifted_masks)
  assert len(orig_images) == len(mask_lists)
  assert len(orig_images) == len(poss_move_masks)
  
  batch_size = orig_images[0].shape[0]
  os.mkdir(os.path.join(output_dir, "itr_" + str(itr)))
  
  gen_image_max = np.max([np.max(x) for x in gen_images])
  shifted_masks = [x - 0.5 for x in shifted_masks]
  #cmap = plt.get_cmap('seismic')
  
  for i in range(batch_size):
    video = []
    for j in range(len(orig_images)):
      video.append(merge(mask_lists[j], orig_images[j], 
                         gen_images[j], shifted_masks[j], poss_move_masks[j], i, gen_image_max))
    clip = mpy.ImageSequenceClip(video, fps=2)
    clip.write_gif(os.path.join(output_dir, "itr_"+str(itr), "All_batch_" + str(i) + ".gif"),
                   verbose=False)
    
  with open(os.path.join(output_dir, "itr_"+str(itr), "shifted_mask.pickle"), "wb") as f:
    cPickle.dump(shifted_masks, f)

def plot_flo(image1, image2, flo, poss_move_mask1, poss_move_mask2, poss_move_maskt, output_dir, itr):
  grey_cmap = plt.get_cmap("Greys")
  batch_size = image1.shape[0]

  h = 2
  w = 3
  img_size = image1.shape[1]
  gap = 3
  
  if not os.path.exists(os.path.join(output_dir, "itr_"+str(itr))):
    os.makedirs(os.path.join(output_dir, "itr_"+str(itr)))
  
  for cnt in range(batch_size):
    img = np.zeros((h * (img_size + gap), w * (img_size + gap), 3))
    for idx in xrange(6):
      i = idx % w
      j = idx // w
      
      if idx == 0:
        tmp = image1[cnt] * 255.0
      elif idx == 1:
        tmp = image2[cnt] * 255.0
      elif idx == 2:
        tmp = flow_to_image(flo[cnt])
      elif idx == 3:
        tmp = grey_cmap(poss_move_mask1[cnt, :, :, 0])[:, :, 0:3] * 255.0
      elif idx == 4:
        tmp = grey_cmap(poss_move_mask2[cnt, :, :, 0])[:, :, 0:3] * 255.0
      else:
        tmp = grey_cmap(poss_move_maskt[cnt, :, :, 0])[:, :, 0:3] * 255.0
      
      img[j*(img_size+gap):j*(img_size+gap)+img_size, i*(img_size+gap):i*(img_size+gap)+img_size, :] = \
          tmp
    
    im = Image.fromarray(img.astype('uint8'))

    im.save(os.path.join(output_dir, "itr_"+str(itr), str(cnt) + ".jpeg"))
  #return img

def plot_eval(image, mask1, mask2, output_dir, itr):
  grey_cmap = plt.get_cmap("Greys")
  batch_size = image.shape[0]

  h = 2
  w = 2
  img_size = image.shape[1]
  gap = 3
  
  if not os.path.exists(os.path.join(output_dir, "itr_"+str(itr))):
    os.makedirs(os.path.join(output_dir, "itr_"+str(itr)))
  
  for cnt in range(batch_size):
    img = np.zeros((h * (img_size + gap), w * (img_size + gap), 3))
    for idx in xrange(3):
      i = idx % w
      j = idx // w
      
      if idx == 0:
        tmp = image[cnt] * 255.0
      elif idx == 1:
        tmp = grey_cmap(mask1[cnt, :, :, 0])[:, :, 0:3] * 255.0
      else:
        tmp = grey_cmap(mask2[cnt, :, :, 0])[:, :, 0:3] * 255.0
      
      img[j*(img_size+gap):j*(img_size+gap)+img_size, i*(img_size+gap):i*(img_size+gap)+img_size, :] = \
          tmp
    
    im = Image.fromarray(img.astype('uint8'))

    im.save(os.path.join(output_dir, "itr_"+str(itr), "eval_" + str(cnt) + ".jpeg"))
    

def plot_grad(var_loss_bg_mask, seg_loss_poss_move_mask, move_poss_move_mask, flo_grad_bg_mask, output_dir, itr):
  batch_size = var_loss_bg_mask.shape[0]
  h = 2
  w = 2
  img_size = var_loss_bg_mask.shape[1]
  gap = 3
  grey_cmap = plt.get_cmap("Greys")
  
  for cnt in range(batch_size):
    fig, ax = plt.subplots(nrows=2,ncols=3, figsize=(20,10))

    heatmap = ax[0, 0].pcolor(flo_grad_bg_mask[cnt, ::-1, :, 0], cmap=grey_cmap)
    fig.colorbar(heatmap, ax=ax[0, 0])
#     
#    heatmap = ax[0, 1].pcolor(-img_grad_poss_move_mask[cnt, ::-1, :, 0], cmap=grey_cmap)
#    fig.colorbar(heatmap, ax=ax[0, 1])
    
    heatmap = ax[1, 0].pcolor(var_loss_bg_mask[cnt, ::-1, :, 0], cmap=grey_cmap)
    fig.colorbar(heatmap, ax=ax[1, 0])
    
#     heatmap = ax[1, 1].pcolor(-img_grad_we_poss_move_mask[cnt, ::-1, :, 0], cmap=grey_cmap)
#     fig.colorbar(heatmap, ax=ax[1, 1])
    
    heatmap = ax[0, 2].pcolor(-move_poss_move_mask[cnt, ::-1, :, 0], cmap=grey_cmap)
    fig.colorbar(heatmap, ax=ax[0, 2])
    
    heatmap = ax[1, 2].pcolor(-seg_loss_poss_move_mask[cnt, ::-1, :, 0], cmap=grey_cmap)
    fig.colorbar(heatmap, ax=ax[1, 2])

    fig.savefig(os.path.join(output_dir, "itr_"+str(itr), "grad_" + str(cnt) + ".png"))
    plt.close(fig)
    
def plot_flo_edge(image1, flo, true_edge, pred_edge,
                 output_dir, itr):
  
  grey_cmap = plt.get_cmap("Greys")
  batch_size = image1.shape[0]

  h = 2
  w = 2
  img_size = image1.shape[1]
  gap = 3
  
  if not os.path.exists(os.path.join(output_dir, "itr_"+str(itr))):
    os.makedirs(os.path.join(output_dir, "itr_"+str(itr)))
  
  for cnt in range(batch_size):
    img = np.zeros((h * (img_size + gap), w * (img_size + gap), 3))
    for idx in xrange(h*w):
      i = idx % w
      j = idx // w
      
      if idx == 0:
        tmp = image1[cnt] * 255.0
      elif idx == 1:
        tmp = flow_to_image(flo[cnt])
      elif idx == 2:
        tmp = grey_cmap(true_edge[cnt, :, :, 0])[:, :, 0:3] * 255.0
      else:
        tmp = grey_cmap(pred_edge[cnt, :, :, 0])[:, :, 0:3] * 255.0
      
      img[j*(img_size+gap):j*(img_size+gap)+img_size, i*(img_size+gap):i*(img_size+gap)+img_size, :] = \
          tmp
    
    im = Image.fromarray(img.astype('uint8'))

    im.save(os.path.join(output_dir, "itr_"+str(itr), str(cnt) + ".jpeg"))


def main():
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
