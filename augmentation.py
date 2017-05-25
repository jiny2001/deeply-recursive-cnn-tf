# coding=utf8
#
# super resolution from
# http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html
#

import os
import numpy as np


import super_resolution_utilty as util

print("Data Augmentation For Training Data")

training_filenames = util.get_files_in_directory("data/ScSR/")
augmented_directory ="data/ScSR2/"
util.make_dir(augmented_directory)

for file_path in training_filenames:
  org_image = util.load_image(file_path)

  _, filename = os.path.split(file_path)
  filename, extension = os.path.splitext(filename)
  
  util.save_image(augmented_directory+filename + extension, org_image)
  ud_image = np.flipud(org_image)
  util.save_image(augmented_directory+filename + "_v" + extension, ud_image)
  lr_image = np.fliplr(org_image)
  util.save_image(augmented_directory+filename + "_h" + extension, lr_image)
  lrud_image = np.flipud(lr_image)
  util.save_image(augmented_directory+filename + "_hv" + extension, lrud_image)

print("\nFinished.")
