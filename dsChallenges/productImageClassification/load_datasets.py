# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 13:43:30 2016

@author: marc
"""
import numpy as np
import os
from scipy import ndimage, misc
from six.moves import cPickle as pickle
from six.moves import range
from IPython.display import display, Image
import matplotlib.pyplot as plt
#%%
import math
from PIL import Image as Img
image_size = 120  # Pixel width and height.
pixel_depth = 3#255.0  # Number of levels per pixel.

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def make_square(im_direct,fill_color=(1,1,1)):
    im = Img.open(im_direct)
    x, y = im.size
    size = max( x, y)
    new_im = Img.new('RGB', (size, size), "white")
    new_im.paste(im, (math.floor((size - x)/2), math.floor((size - y)/2) ))
    return new_im

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size,pixel_depth),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = ndimage.imread(image_file).astype(float)
      if len(image_data.shape)<3:
         img = Img.open(image_file)
         x, y = img.size
         #size = max( x, y)
         rgbimg = img.convert("RGB")
         #rgbimg.show()
         #rgbimg = rgbimg.paste(img)
         image_data = np.array(rgbimg)
         #print(image_file+" "+str(image_data.shape)+" "+str(img.size)+" "+str(rgbimg.size))
         #rgbimg.show()
      if image_data.shape != (image_size, image_size,pixel_depth):
        #print(len(image_data.shape))
#        if len(image_data.shape)<3:
#            display(Image(image_file))
#            print(ndimage.imread(image_file).astype(float).shape)
        if image_data.shape[0]!=image_data.shape[1]:#image not square
            image_squared = make_square(image_file)
            image_data = np.array(image_squared)
        image_data = misc.imresize(image_data, (image_size, image_size))    
        #raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :,:] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
def maybe_pickle(folder_name,data_folders, min_num_images_per_class, force=True):
  dataset_names = []
  if len(data_folders)>=10:
      data_folders = data_folders[1:]
  for folder in data_folders:
    #print(folder)
    folder_direct = folder_name+folder
    #print(folder_direct)
    set_filename = folder_direct + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) or folder_direct and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder_direct, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)
  
  return dataset_names
  
#display(Image(filename="train/101-1065/0a1c6ae790ee2e83d6a2e84cc156a8c1.jpg"))
image_produit = Image(filename="train/101-1065/0a1c6ae790ee2e83d6a2e84cc156a8c1.jpg")
train_folder = 'train/'
valid_folder = 'val/'
# We get all file names
fn = os.listdir("val/101-1065/")
# Display first 20 images 

#for file in fn[:20]:
#    path = 'val/101-1065/' + file
#    im=Img.open(path)
#    display(Image(path))
#    print(im.size)
valid_datasets = maybe_pickle(valid_folder,os.listdir(valid_folder), 1)
valid_datasets = maybe_pickle(valid_folder,os.listdir(valid_folder), 1)
#%%
train_datasets = maybe_pickle(train_folder,os.listdir(train_folder), 100)

#%%

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes
    
  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class
                    
        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise
    
  return valid_dataset, valid_labels, train_dataset, train_labels

def merge_datasets2(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = None,None
  train_dataset, train_labels = None,None
  #vsize_per_class = valid_size // num_classes
  #tsize_per_class = train_size // num_classes
    
  #start_v, start_t = 0, 0
  #end_v, end_t = vsize_per_class, tsize_per_class
  #end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):       
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is None:
            valid_dataset = letter_set
            valid_labels = np.ndarray(letter_set.shape[0], dtype=np.dtype(('U10', 1)))
            valid_labels[:] = label
        else:
          valid_dataset = np.append(valid_dataset, letter_set, axis=0)
          #valid_dataset[start_v:end_v, :, :] = valid_letter
          current_label = np.ndarray(letter_set.shape[0], dtype=np.dtype(('U10', 1)))
          valid_labels[start_v:end_v] = np.append(valid_labels, current_label, axis=0)
          
  

  valid_dataset,valid_labels = shuffle(valid_dataset,valid_labels, random_state=0)
  train_dataset, valid_labels, train_labels, valid_labels = train_test_split(X, y, test_size=.1, random_state=0)#
  return valid_dataset, valid_labels, train_dataset, train_labels
            
            
train_size = 60000
valid_size = 200
test_size = 400

test_dataset, test_labels, train_dataset, train_labels = merge_datasets(
  train_datasets, train_size, test_size)


print('Training:', train_dataset.shape, train_labels.shape)
print('Test:', test_dataset.shape, test_labels.shape)
#print('Validation:', valid_dataset.shape, valid_labels.shape)

#%%
valid_size = 200
_, _, valid_dataset, valid_labels = merge_datasets(valid_datasets, valid_size)
print('Validation:', valid_dataset.shape, valid_labels.shape)

#%%
for numbers in range(5):
    letter = train_dataset[numbers,:,:]
    plt.imshow(letter, extent=(letter.min(), letter.max(), letter.max(), letter.min()),
               interpolation='nearest', cmap=cm.gist_rainbow)
    plt.show()

#%%


def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels
train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

#%%

for numbers in range(5):
    letter = train_dataset[numbers,:,:]
    plt.imshow(letter, extent=(letter.min(), letter.max(), letter.max(), letter.min()),
               interpolation='nearest', cmap=cm.gist_rainbow)
    plt.show()
#%% save datasets in one pickle file
    pickle_file = 'data.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
  
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

#%% Measure overlap between training set and validation set or test set
overlap_valid = 0;
valid_size_loop = valid_size
valid_dataset_same = valid_dataset
for i in range(train_size):
    for j in range(valid_size_loop):
        maybe_same = train_dataset[i,:,:]==valid_dataset_same[j,:,:]
        if maybe_same.all():
            overlap_valid +=1
            valid_size_loop -=1
            valid_dataset_same = np.delete(valid_dataset_same,j,0)
            break

overlap_test = 0;
test_size_loop = test_size
test_dataset_same = test_dataset
for i in range(train_size):
    for j in range(test_size_loop):
        maybe_same = train_dataset[i,:,:]==test_dataset_same[j,:,:]
        if maybe_same.all():
            overlap_test +=1
            test_size_loop -=1
            test_dataset_same = np.delete(test_dataset_same,j,0)
            break
            
            
#%% Logistic regression
from sklearn import linear_model
logistic_classifier = linear_model.LogisticRegression(C=1)
a_train,b_train,c_train = train_dataset.shape
train_dataset_2D = np.reshape(train_dataset, (a_train,b_train*c_train))
a_test,b_test,c_test = test_dataset.shape
test_dataset_2D = np.reshape(test_dataset, (a_test,b_test*c_test))
a_valid,b_valid,c_valid = valid_dataset.shape
valid_dataset_2D = np.reshape(valid_dataset, (a_valid,b_valid*c_valid))
logistic_classifier.fit(train_dataset_2D,train_labels)
logistic_classifier.predict(valid_dataset_2D)
logistic_classifier.score(test_dataset_2D,test_labels)

    
