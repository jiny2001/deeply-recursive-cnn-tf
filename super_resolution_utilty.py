# coding=utf8

"""
Deeply-Recursive Convolutional Network for Image Super-Resolution
Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html

Test implementation utility
Author: Jin Yamanaka
"""

from __future__ import division
import shutil
import datetime
import os
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np
from scipy import misc
import scipy.stats as stats
from PIL import Image


# utilities for save / load

test_datasets = {
  "set5": ["Set5", 0, 5],
  "set14": ["Set14", 0, 14],
  "bsd100": ["BSD100_SR", 0, 100],
  "urban100": ["Urban100_SR", 0, 100],
  "test": ["Set5", 0, 1]
}


class LoadError(Exception):
  
  def __init__(self, message):
    self.message = message

    
def make_dir(directory):
  
  if not os.path.exists(directory):
      os.makedirs(directory)


def get_files_in_directory(path):
  
  file_list = [path + f for f in listdir(path) if isfile(join(path, f))]
  return file_list


def remove_generic(path, __func__):
  
  try:
    __func__(path)
  except OSError, (error_no, error_str):
    print "Error removing %(path)s, %(error)s" % {'path': path, 'error': error_str}


def clean_dir(path):
  
  if not os.path.isdir(path):
    return

  files = os.listdir(path)
  for x in files:
    full_path = os.path.join(path, x)
    if os.path.isfile(full_path):
      f = os.remove
      remove_generic(full_path, f)
    elif os.path.isdir(full_path):
      clean_dir(full_path)
      f = os.rmdir
      remove_generic(full_path, f)


def save_image(filename, image):

  if len(image.shape) >= 3 and image.shape[2] == 1:
    image = image.reshape(image.shape[0], image.shape[1])

  directory = os.path.dirname(filename)
  if directory != "" and not os.path.exists(directory):
    os.makedirs(directory)
    
  misc.imsave(filename, image)

  print 'Saved [%s]' % filename


def save_image_data(filename, image):

    directory = os.path.dirname(filename)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    np.save(filename, image)
    print 'Saved [%s]' % filename

    if len(image.shape) ==3 and image.shape[2] == 1:
      image2 = image.reshape(image.shape[0], image.shape[1])
    misc.imsave(filename, image2)


def convert_rgb_to_y(image, jpeg_mode=True, max_value=255):

  if len(image.shape) <= 2 or image.shape[2] == 1:
    return image
  
  y_image = np.zeros([image.shape[0], image.shape[1], 1])
  if jpeg_mode:
    for i in xrange(image.shape[0]):
      for j in xrange(image.shape[1]):
        y_image[i, j, 0] = 0.299 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 2]
  else:
    for i in xrange(image.shape[0]):
      for j in xrange(image.shape[1]):
        y_image[i, j, 0] = (16.0 * max_value + 65.481 * image[i, j, 0] + 128.553 * image[i, j, 1] + 24.966 * image[i, j, 2] ) / 256.0

  return y_image


def convert_rgb_to_ycbcr(image, jpeg_mode=True, max_value=255):
  
  if len(image.shape) < 2 or image.shape[2] == 1:
    return image
  
  ycbcr_image = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
  if jpeg_mode:
    for i in xrange(image.shape[0]):
      for j in xrange(image.shape[1]):
        ycbcr_image[i, j, 0] = 0.299 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 2]
        ycbcr_image[i, j, 1] = -0.169 * image[i, j, 0] - 0.331 * image[i, j, 1] + 0.500 * image[i, j, 2] + max_value / 2
        ycbcr_image[i, j, 2] = 0.500 * image[i, j, 0] - 0.419 * image[i, j, 1] - 0.081 * image[i, j, 2] + max_value / 2
  else:
    for i in xrange(image.shape[0]):
      for j in xrange(image.shape[1]):
        ycbcr_image[i, j, 0] = ( 16.0 * max_value + 65.481 * image[i, j, 0] + 128.553 * image[i, j, 1] + 24.966 * image[i, j, 2] ) / 256.0
        ycbcr_image[i, j, 1] = (128.0 * max_value - 37.945 * image[i, j, 0] - 74.494 * image[i, j, 1] + 112.439 * image[i, j, 2] ) / 256.0
        ycbcr_image[i, j, 2] = (128.0 * max_value + 112.439 * image[i, j, 0] - 94.154 * image[i, j, 1] - 18.285 * image[i, j, 2] ) / 256.0
  
  return ycbcr_image


def convert_y_and_cbcr_to_rgb(y_image, cbcr_image, jpeg_mode=True, max_value=255.0):
  
  if len(y_image.shape) <= 2:
    y_image = y_image.reshape[y_image.shape[0], y_image.shape[1], 1]

  if len(y_image.shape) == 3 and y_image.shape[2] == 3:
    y_image = y_image[:,:,0:1]

  ycbcr_image = np.zeros([y_image.shape[0], y_image.shape[1], 3])
  ycbcr_image[:, :, 0] = y_image[:, :, 0]
  ycbcr_image[:, :, 1:3] = cbcr_image[:, :, 0:2]

  return convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=jpeg_mode, max_value=max_value)


def convert_ycbcr_to_rgb(ycbcr_image, jpeg_mode=True, max_value=255.0):

  rgb_image = np.zeros([ycbcr_image.shape[0], ycbcr_image.shape[1], 3])
  center = max_value / 2
  if jpeg_mode:
    for i in xrange(ycbcr_image.shape[0]):
      for j in xrange(ycbcr_image.shape[1]):
        rgb_image[i, j, 0] = ycbcr_image[i, j, 0] + 1.402 * (ycbcr_image[i, j, 2] - center)
        rgb_image[i, j, 1] = ycbcr_image[i, j, 0] - 0.344 * (ycbcr_image[i, j, 1] - center) - 0.714 * (
        ycbcr_image[i, j, 2] - center)
        rgb_image[i, j, 2] = ycbcr_image[i, j, 0] + 1.772 * (ycbcr_image[i, j, 1] - center)
  else:
    for i in xrange(ycbcr_image.shape[0]):
      for j in xrange(ycbcr_image.shape[1]):
        y = ycbcr_image[i, j, 0] - 16 * max_value / 256
        cb = ycbcr_image[i, j, 1] - center
        cr = ycbcr_image[i, j, 2] - center
        rgb_image[i, j, 0] = (y / 219.0 + 0.701 * cr / 112) * max_value
        rgb_image[i, j, 1] = (y / 219.0 - 0.886 * 0.114 / (112 * 0.587) * cb - 0.701 * 0.299 / (112 * 0.587) * cr) * max_value
        rgb_image[i, j, 2] = (y / 219.0 + 0.886 / 112.0 * cb) * max_value

  return rgb_image


def set_image_alignment(image, alignment):
  
  alignment = int(alignment)    # I don't like this...
  width, height = image.shape[1], image.shape[0]
  width = (width // alignment) * alignment
  height = (height // alignment) * alignment
  if image.shape[1] != width or image.shape[0] != height:
    return image[:height, :width, :]
  
  return image


def resize_image_by_bicubic(image, scale):
  
  size = [int(image.shape[0] * scale), int(image.shape[1] * scale)]
  image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
  tf_image = tf.image.resize_bicubic(image, size=size)
  image = tf_image.eval()
  return image.reshape(image.shape[1], image.shape[2], image.shape[3])

def resize_image_by_scipy_bicubic(image, scale):

  if len(image.shape)==3 and image.shape[2] == 1:
    image = image.reshape(image.shape[0], image.shape[1])
    image = misc.imresize(image, float(scale), interp="bicubic")
    return image.reshape(image.shape[0], image.shape[1], 1)
  else:
    return misc.imresize(image, float(scale), interp="bicubic")

def resize_image_by_nnr(image, scale):

  if len(image.shape)==3 and image.shape[2] == 1:
    image = image.reshape(image.shape[0], image.shape[1])
    image = misc.imresize(image, float(scale), interp="nearest")
    return image.reshape(image.shape[0], image.shape[1], 1)
  else:
    return misc.imresize(image, float(scale), interp="nearest")


def resize_image_by_pil_bicubic(image, scale):
  
  width, height = image.shape[1], image.shape[0]
  if len(image.shape)==3 and image.shape[2] == 3:
    image = Image.fromarray(image,"RGB")
    image = image.resize([int(height*scale), int(width*scale)], resample=Image.BICUBIC)
    image = np.asarray(image)
  else:
    image = Image.fromarray(image.reshape(height, width))
    image = image.resize([int(height * scale), int(width * scale)], resample=Image.BICUBIC)
    image = np.asarray(image)
    image = image.reshape(image.shape[0], image.shape[1], 1)
  return image


def load_image(filename, width=0, height=0, channels=0, alignment=0):
  
  if not os.path.isfile(filename):
    raise LoadError("File not found")
  image = misc.imread(filename)

  if len(image.shape) == 2:
    image = image.reshape(image.shape[0], image.shape[1], 1)
  if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
    raise LoadError("Attributes mismatch")
  if channels != 0 and image.shape[2] != channels:
    raise LoadError("Attributes mismatch")
  if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
    raise LoadError("Attributes mismatch")

  print "Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2])
  return image


def load_image_data(filename, width=0, height=0, channels=0, alignment=0):

    if not os.path.isfile(filename+".npy"):
        raise LoadError("File not found")
    image = np.load(filename+".npy")

    if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
        raise LoadError("Attributes mismatch")
    if channels != 0 and image.shape[2] != channels:
        raise LoadError("Attributes mismatch")
    if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
        raise LoadError("Attributes mismatch")

    print "Cache Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2])
    return image


def load_input_image_with_cache(cache_dir, org_filename, channels=1, alignment=0, scale=1,
                     convert_ycbcr=True, jpeg_mode=False, max_value=255.0):

  if cache_dir is None or cache_dir is "":
    return load_input_image(org_filename, channels=channels, alignment=alignment, scale=scale)

  filename, extension = os.path.splitext(org_filename)
  if scale != 1.0:
    filename += "_%1.0f" % scale
  if channels == 1:
    filename += "_Y"

  cache_filename = cache_dir + "/" + filename + extension
  try:
    image = load_image_data(cache_filename, channels=channels)
  except LoadError:
    image = load_input_image(org_filename, channels=channels, alignment=scale, scale=scale,
                             convert_ycbcr=convert_ycbcr, jpeg_mode=False, max_value=max_value)
    save_image_data(cache_filename, image)

  return image


def load_input_image(filename, width = 0, height = 0, channels=1, alignment=0, scale=1,
                     convert_ycbcr=True, jpeg_mode=False, max_value=255.0):

  image = load_image(filename)

  if width != 0 and height != 0:
    if image.shape[0] != height or image.shape[1] != width:
      x = (image.shape[1] - width) // 2
      y = (image.shape[0] - height) // 2
      image = image[y: y + height, x: x + width, :]

  if alignment > 1:
    image = set_image_alignment(image, alignment)

  if scale != 1:
    image = resize_image_by_bicubic(image, 1.0 / scale)
    image = resize_image_by_bicubic(image, scale)
    
  if convert_ycbcr:
    image = convert_rgb_to_ycbcr(image, jpeg_mode=jpeg_mode)
  
  if channels == 1 and image.shape[2] > 1:
    image = image[:, :, 0:1].copy()   # use copy() since after the step we use stride_tricks.as_strided().

  if max_value != 255.0:
      image = np.multiply(image, max_value / 255.0)

  return image


def get_split_images(image, window_size, stride=None):
  
  if len(image.shape) == 3 and image.shape[2] == 1:
    image = image.reshape(image.shape[0], image.shape[1])

  window_size = int(window_size)
  size = image.itemsize   # byte size of each value
  height, width = image.shape
  if stride is None:
    stride = window_size
  else:
    stride = int(stride)

  new_height = 1 + (height - window_size) // stride
  new_width = 1 + (width - window_size) // stride

  shape = (new_height, new_width, window_size, window_size)
  strides = size * np.array([width * stride, stride, width, 1])
  windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
  windows = windows.reshape( windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3],1)

  return windows


# utilities for building graphs

def conv2d(x, w, stride, name=""):
  
  return tf.nn.conv2d(x, w, strides=[stride, stride, 1, 1], padding='SAME', name=name + "_conv")


def conv2d_with_bias_and_relu(x, w, stride, bias, name=""):
  
  conv = conv2d(x, w, stride, name)
  return tf.nn.relu( tf.add(conv, bias, name=name+"_add"), name=name + "_relu")


def weight(shape, stddev=0.01, name=None):
  
  initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev)
  if name is None:
    return tf.Variable(initial)
  else:
    return tf.Variable(initial, name=name)


def diagonal_cnn_weight(shape, stddev=0.0, name=None):
  
  if stddev == 0.0:
    initial = np.zeros(shape, dtype=float)
  else:
    initial = np.random.normal(0, stddev, shape)
    initial = stats.threshold(initial, threshmin=-2 * stddev, threshmax=2 * stddev)

  if len(shape) == 4:
    i = shape[0] // 2
    j = shape[1] // 2
    for k in xrange(min(shape[2], shape[3])):
      initial[i][j][k][k] = 1.0

  initial = tf.cast(initial, tf.float32)
  if name is None:
    return tf.Variable(initial)
  else:
    return tf.Variable(initial, name=name)


def bias(shape, initial_value=0.0, name=None):

  if name is None:
    initial = tf.constant(initial_value, dtype=tf.float32, shape=shape)
  else:
    initial = tf.constant(initial_value, dtype=tf.float32, shape=shape, name=name)
  return tf.Variable(initial)


# utilities for logging -----

def add_summaries(name, var, stddev=True, mean=False, max=False, min=False):

  with tf.name_scope("summaries"):

    mean_var = tf.reduce_mean(var)
    if mean:
      tf.scalar_summary('mean/' + name, mean_var)

    if stddev:
      with tf.name_scope('stddev'):
        stddev_var = tf.sqrt(tf.reduce_sum(tf.square(var - mean_var)))
        tf.scalar_summary('stddev/' + name, stddev_var)

    if max:
      tf.scalar_summary('max/' + name, tf.reduce_max(var))
    
    if min:
      tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)


def get_now_date():
  
  d = datetime.datetime.today()
  return '%s/%s/%s %s:%s:%s' % (d.year, d.month, d.day, d.hour, d.minute, d.second)


def compute_mse(image1, image2, border_size = 0):
  
  if len(image1.shape)==2:
    image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
  if len(image2.shape)==2:
    image2 = image2.reshape(image2.shape[0], image2.shape[1], 1)

  if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1] or image1.shape[2] != image2.shape[2]:
    return None

  if image1.dtype == np.uint8:
    image1 = image1.astype(np.double)
  if image2.dtype == np.uint8:
    image2 = image2.astype(np.double)

  mse = 0.0
  for i in xrange(border_size, image1.shape[0] - border_size):
    for j in xrange(border_size, image1.shape[1] - border_size):
      for k in xrange(image1.shape[2]):
        error = image1[i, j, k] - image2[i, j, k]
        mse += error * error
  
  return mse / ((image1.shape[0] - 2 * border_size) * (image1.shape[1] - 2 * border_size) * image1.shape[2])


def print_CNN_weight(tensor):
  
  print "Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape()))
  weight = tensor.eval()
  for i in xrange(weight.shape[3]):
    values = ""
    for x in xrange(weight.shape[0]):
      for y in xrange(weight.shape[1]):
        for c in xrange(weight.shape[2]):
          values += "%2.3f " % weight[y][x][c][i]
    print values
  print "\n"


def print_CNN_bias(tensor):
  print "Tensor[%s] shape=%s" % (tensor.name, str(tensor.get_shape()))
  bias = tensor.eval()
  values = ""
  for i in xrange(bias.shape[0]):
    values += "%2.3f " % bias[i]
  print values + "\n"
  

def get_test_filenames(data_folder, dataset, scale):

  test_folder = data_folder + "/" + test_datasets[dataset][0] + "/image_SRF_%d/" % scale

  test_filenames = []
  for i in xrange(test_datasets[dataset][1], test_datasets[dataset][2]):
    test_filenames.append(test_folder + "img_%03d_SRF_%d_HR.png" % (i + 1, scale))
  
  return test_filenames


def build_test_filenames(data_folder, dataset, scale):
  
  test_filenames = []

  if dataset == "all":
    for test_dataset in test_datasets:
      test_filenames += get_test_filenames(data_folder, test_dataset, scale)
  else:
    test_filenames += get_test_filenames(data_folder, dataset, scale)

  return test_filenames


# utility for extracting target files from datasets
def main():
  
  flags = tf.app.flags
  FLAGS = flags.FLAGS
  
  flags.DEFINE_string("org_data_folder", "org_data", "Folder for original datasets")
  flags.DEFINE_string("test_set", "all", "Test dataset. set5, set14, bsd100, urban100 or all are available")
  flags.DEFINE_integer("scale", 2, "Scale for Super Resolution (can be 2 or 4)")
  
  test_filenames = build_test_filenames(FLAGS.org_data_folder, FLAGS.test_set, FLAGS.scale)

  for filename in test_filenames:
    target_filename = "data/" + filename
    print "[%s] > [%s]" % (filename, target_filename)
    if not os.path.exists(os.path.dirname(target_filename)):
      os.makedirs(os.path.dirname(target_filename))
    shutil.copy(filename, target_filename)
  
  print "OK."


if __name__ == '__main__':
  main()
