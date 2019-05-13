import numpy as np
import tensorflow as tf

from ... import tflib as lib
from ...tflib.ops import linear
from ...tflib.ops import conv2d
from ...tflib.ops import batchnorm
from ...tflib.ops import deconv2d

MODE = 'dcgan' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    layers = []

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    layers.append(output)
    output = tf.reshape(output, [-1, 4, 4, 4*DIM])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    layers.append(output)
    output = tf.nn.relu(output)

    output = output[:,:7,:7,:]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
    layers.append(output)

    logits = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    output = tf.nn.sigmoid(logits)

    return tf.reshape(logits, [-1, OUTPUT_DIM]), tf.reshape(output, [-1, OUTPUT_DIM]), layers

def Discriminator(inputs, keep_prob):
    output = tf.reshape(inputs, [-1, 28, 28, 1])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)

    return tf.reshape(output, [-1])
