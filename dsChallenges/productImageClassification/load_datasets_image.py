# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:43:30 2016

@author: marc
"""
import numpy as np
import os
from scipy import ndimage, misc
#from six.moves import cPickle as pickle
#from six.moves import range
#from IPython.display import display, Image
#import matplotlib.pyplot as plt
import math
from PIL import Image as Img
#%% load the images of each folder and make them square
image_size = 140  # Pixel width and height.
pixel_depth = 3 # Number of levels per pixel for RGB its 3.

def make_square(im_direct,fill_color=(1,1,1)):
    im = Img.open(im_direct)
    x, y = im.size
    size = max( x, y)
    new_im = Img.new('RGB', (size, size), "white")
    new_im.paste(im, (math.floor((size - x)/2), math.floor((size - y)/2) ))
    return new_im

def load_image(folder, min_num_images):
  """Load the data for a single label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size,pixel_depth),
                         dtype=np.float32)#channels last format 
  label = np.ndarray(shape=(len(image_files),1),dtype=np.dtype(('U10', 1))  )
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      if num_images<800:
          image_data = ndimage.imread(image_file).astype(float)
          if len(image_data.shape)<3:#grayscale image
             img = Img.open(image_file)
             rgbimg = img.convert("RGB")
             image_data = np.array(rgbimg)
          if image_data.shape != (image_size, image_size,pixel_depth):
            if image_data.shape[0]!=image_data.shape[1]:#image not square
                image_squared = make_square(image_file)
                image_data = np.array(image_squared)
            image_data = misc.imresize(image_data, (image_size, image_size)) #resize the image   
          dataset[num_images, :, :,:] = image_data
          label[num_images]=folder.split("/")[1]
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  if dataset.shape[0]>800:
      dataset = dataset[:800,:,:,:]
      label = label[:800]
  print('Smaller dataset tensor:', dataset.shape)
  return dataset,label
        
def load_images(folder_name,data_folders, min_num_images_per_class, force=True):
  dataset,label = None,None
  if len(data_folders)>=10:
      data_folders = data_folders[1:]
  for folder in data_folders:
    folder_direct = folder_name+folder
    current_dataset,current_label = load_image(folder_direct, min_num_images_per_class)
    if dataset is None :
        dataset = current_dataset
        label = current_label
    else:
        dataset = np.append(dataset,current_dataset,axis=0)
        label = np.append(label,current_label,axis=0)
  return dataset,label
  
train_folder = 'train/'
valid_folder = 'val/'

valid_dataset,valid_label = load_images(valid_folder,os.listdir(valid_folder), 1)
train_dataset,train_label = load_images(train_folder,os.listdir(train_folder), 1)

#%% save train and validation datasets into files
np.save("train_label", train_label)
np.save("valid_label", valid_label)
np.save("train_dataset", train_dataset)
np.save("valid_dataset", valid_dataset)


#%%
def load_image_test(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size,pixel_depth),
                          dtype=np.float32)#channels last format 
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = ndimage.imread(image_file).astype(float)
      if len(image_data.shape)<3:
         img = Img.open(image_file)
         x, y = img.size
         rgbimg = img.convert("RGB")
         image_data = np.array(rgbimg)
      if image_data.shape != (image_size, image_size,pixel_depth):
        if image_data.shape[0]!=image_data.shape[1]:#image not square
            image_squared = make_square(image_file)
            image_data = np.array(image_squared)
        image_data = misc.imresize(image_data, (image_size, image_size))   
      dataset[num_images, :, :,:] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  print('Smaller dataset tensor:', dataset.shape)
  return dataset

test_folder = "test/"
test_dataset = load_image_test(test_folder, 1)

#%%save test dataset into files
np.save("test_dataset", test_dataset)


    
