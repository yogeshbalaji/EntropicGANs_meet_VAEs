import tensorflow as tf
import numpy as np
import pkg.tflib as lib
import pkg.tflib.mnist
import pkg.tflib.cifar10
import pkg.tflib.save_images
import os


FLAGS = tf.flags.FLAGS

def try_restore_checkpoint(saver):
    path = FLAGS.savedir
    print('Loading from {}'.format(path))
    latest_checkpoint = tf.train.latest_checkpoint(path)
    if not latest_checkpoint:
      print('No checkpoint found')
      return False
    else:
      print('Checkpoint Found')
      if FLAGS.restore_checkpoint:
        print('Restore checkpoint')
        saver.restore(session, latest_checkpoint)
      return True

def save_checkpoint(saver, step, session):
    filename = FLAGS.savedir + '/model.ckpt-' + str(step)
    print('Saving checkpoint {}'.format(filename))
    saver.save(session, filename)

def generate_image(frame, samples):
    if len(samples)>128:
      samples = samples[:128]
    if FLAGS.dataset == 'mnist':
      samples = samples.reshape((128, 28, 28))
    elif FLAGS.dataset == 'cifar10':
      samples = ((samples+1.)*(255./2)).astype('int32')
      samples = samples.reshape((128, 32, 32, 3)).transpose((0,3,1,2))

    if not tf.gfile.Exists(FLAGS.savedir):
        tf.gfile.MkDir(FLAGS.savedir)
    lib.save_images.save_images(
        samples, 
        '{}/samples_{}_{}.png'.format(FLAGS.savedir, FLAGS.savename,frame)
    )



def data_loader():
  # Dataset iterator
  if FLAGS.dataset == 'mnist': 
    train_gen, dev_gen, test_gen = lib.mnist.load(FLAGS.batch_size, FLAGS.batch_size)
    def inf_train_gen():
      while True:
          for images,targets in train_gen():
              yield images, targets
  
  elif FLAGS.dataset == 'cifar10':
    DATA_DIR = 'cifar-10-batches-py' # Specify CIFAR-10 path
    train_gen, test_gen = lib.cifar10.load(FLAGS.batch_size, data_dir=DATA_DIR)
    def inf_train_gen():
      while True:
          for images,targets in train_gen():
              yield images, targets

  return inf_train_gen(), test_gen


def data_set():
  # Dataset iterator
  if FLAGS.dataset == 'mnist': 
    train_gen, dev_gen, test_gen = lib.mnist.load(50000, 10000)
    def inf_train_gen():
      while True:
          for images, targets in train_gen():
              yield images, targets
    def inf_test_gen():
      while True:
          for images,targets in test_gen():
              yield images, targets
  
  elif FLAGS.dataset == 'cifar10':
    DATA_DIR = 'cifar-10-batches-py' # Specify CIFAR-10 path
    train_gen, test_gen = lib.cifar10.load(FLAGS.batch_size, data_dir=DATA_DIR)
    def inf_train_gen():
      while True:
          for images,targets in train_gen():
              yield images, targets
    def inf_test_gen():
      while True:
          for images,targets in test_gen():
              yield images, targets

  return inf_train_gen(), inf_test_gen()


