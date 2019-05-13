import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import cPickle as pickle
import os

FLAGS = tf.flags.FLAGS

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})
_since_last_flush_ordered_key = []
_iter = [0]

def tick():
	_iter[0] += 1

def plot(name, value):
  global _since_last_flush_ordered_key
  if not (name in _since_last_flush_ordered_key):
    _since_last_flush_ordered_key.append(name)
  _since_last_flush[name][_iter[0]] = value


def check_dir(path):
  if not tf.gfile.Exists(path):
    tf.gfile.MkDir(path)
  #if not os.path.exists(path):
  #  os.makedirs(path)

def flush():
  check_dir(FLAGS.savedir)

  prints = []
  global _since_last_flush_ordered_key
  for name in _since_last_flush_ordered_key:
    vals = _since_last_flush[name]
    prints.append("{}\t{}".format(name, np.mean(vals.values())))
    _since_beginning[name].update(vals)
    
    x_vals = np.sort(_since_beginning[name].keys())
    y_vals = [_since_beginning[name][x] for x in x_vals]
    
    plt.clf()
    plt.plot(x_vals, y_vals)
    plt.xlabel('iteration')
    plt.ylabel(name)
    save_name = name.replace(' ', '_')+'.jpg'
    # plt.savefig(save_name)
    plt.savefig('{}/imgs/{}'.format(FLAGS.savedir, save_name))
  
  print "iter {}\t{}".format(_iter[0], "\t".join(prints))
  _since_last_flush.clear()
  _since_last_flush_ordered_key = []

  LOG_FILE = '{}/log.pkl'.format(FLAGS.savedir)
  with open(LOG_FILE, 'wb') as f:
    pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
