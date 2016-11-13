# coding=utf8

"""
Deeply-Recursive Convolutional Network for Image Super-Resolution
by Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea

Paper: http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.html

Test implementation using TensorFlow library.

Author: Jin Yamanaka
Many thanks for: Masayuki Tanaka and Shigesumi Kuwashima
Project: https://github.com/

note:
github repository now doesn't contain x3 or x4 data et for Urban100 and BSD100
"""

import tensorflow as tf
import super_resolution as sr
import super_resolution_utilty as util

flags = tf.app.flags
FLAGS = flags.FLAGS

# Model
flags.DEFINE_float("initial_lr", 0.0005, "Initial learning rate")
flags.DEFINE_float("lr_decay", 0.2, "Learning rate decay rate when it does not reduced during specific epoch")
flags.DEFINE_integer("lr_decay_epoch", 5, "Decay learning rate when loss does not decrease")
flags.DEFINE_float("beta1", 0.1, "Beta1 form adam optimizer")
flags.DEFINE_float("momentum", 0.9, "Momentum for momentum optimizer and rmsprop optimizer")
flags.DEFINE_integer("feature_num", 256, "Number of CNN Filters")
flags.DEFINE_integer("cnn_size", 3, "Size of CNN filters")
flags.DEFINE_integer("inference_depth", 16, "Number of recurrent CNN filters")
flags.DEFINE_integer("batch_num", 64, "Number of mini-batch images for training")
flags.DEFINE_integer("batch_size", 41, "Image size for mini-batch")
flags.DEFINE_integer("stride_size", 21, "Stride size for mini-batch")
flags.DEFINE_string("optimizer", "adam", "Optimizer: can be [gd, adadelta, adagrad, adam, momentum, rmsprop]")
flags.DEFINE_float("loss_alpha", 0.5, "Initial loss-alpha value. Don't use intermediate outputs when 0.")
flags.DEFINE_integer("loss_alpha_zero_epoch", 100, "Decrease loss-alpha to zero by this epoch")
flags.DEFINE_float("loss_beta", 100, "Loss-beta for r️educing the divergence of parameter values")
flags.DEFINE_float("weight_dev", 0.01, "Initial weight stddev")

# Image Processing
flags.DEFINE_integer("scale", 2, "Scale for Super Resolution (can be 2 or 4)")
flags.DEFINE_float("max_value", 255.0, "For normalize image pixel value")
flags.DEFINE_integer("channels", 1, "Using num of image channels. Use YCbCr when channels=1.")

# Training or Others
flags.DEFINE_boolean("is_training", True, "Train model with 91 standard images")
flags.DEFINE_string("dataset", "set5", "Test dataset. [set5, set14, bsd100, urban100, all, test] are available")
flags.DEFINE_integer("evaluate_step", 10, "steps for evaluation")
flags.DEFINE_integer("save_step", 500, "steps for saving learned model")
flags.DEFINE_float("end_lr", 1e-5, "Training end learning rate")
flags.DEFINE_string("checkpoint_dir", "model", "Directory for checkpoints")
flags.DEFINE_string("cache_dir", "", "Directory for caching image data. If specified, build image cache")
flags.DEFINE_string("data_dir", "data", "Directory for test/train images")
flags.DEFINE_boolean("load_model", False, "Load saved model before start")

# Debugging or Logging
flags.DEFINE_string("output_dir", "output", "Directory for output test images")
flags.DEFINE_string("log_dir", "tf_log", "Directory for tensorboard log")
flags.DEFINE_boolean("debug", False, "Display each calculated MSE and weight variables")
flags.DEFINE_boolean("initialise_log", True, "Clear all tensorboard log before start")
flags.DEFINE_boolean("visualize", True, "Save loss and graph data")


def main(_):
  
  print "Super Resolution (tensorflow version:%s)" % tf.__version__
  print "%s\n" % util.get_now_date()

  model_name = "model_F%d_D%d_LR%f" % (FLAGS.feature_num, FLAGS.inference_depth, FLAGS.initial_lr)
  model = sr.SuperResolution(FLAGS, model_name=model_name)

  test_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.dataset, FLAGS.scale)
  if FLAGS.is_training:
    if FLAGS.dataset == "test":
      training_filenames = util.build_test_filenames(FLAGS.data_dir, FLAGS.dataset, FLAGS.scale)
    else:
      training_filenames =  util.get_files_in_directory(FLAGS.data_dir + "/ScSR/")

    print "Loading and building cache images..."
    model.load_datasets(FLAGS.cache_dir, training_filenames, test_filenames, FLAGS.batch_size, FLAGS.stride_size,
                          FLAGS.scale)
  else:
    FLAGS.load_model = True

  model.build_embedding_graph()
  model.build_inference_graph()
  model.build_reconstruction_graph()
  model.build_optimizer()
  model.init_all_variables(load_initial_data=FLAGS.load_model)

  if FLAGS.is_training:
    train(training_filenames, test_filenames, model)
  
  psnr = 0
  for filename in test_filenames:
    mse = model.do_super_resolution_for_test(filename, FLAGS.output_dir, FLAGS.scale)
    psnr += model.get_psnr(mse)

  print "\n%s Final PSNR:%f" % (util.get_now_date(), psnr / len(test_filenames))
  
  
def train(training_filenames, test_filenames, model):

  step = 0
  while model.lr > FLAGS.end_lr:
  
    step += 1
    model.build_training_batch()
    model.run_train_step()
      
    if step % FLAGS.evaluate_step == 0:
      mse = model.evaluate(step)
      model.print_status(step, mse)

    if step > 0 and step % FLAGS.save_step == 0:
      model.save()

  model.end_train_step(step)
  
  model.save()
  if FLAGS.debug:
    model.print_weight_variables()


if __name__ == '__main__':
  tf.app.run()
