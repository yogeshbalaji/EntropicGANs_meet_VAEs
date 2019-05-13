import os, sys
sys.path.append(os.getcwd())

import tensorflow as tf
import pkg.tflib as lib
import pkg.tflib.ops.linear
import pkg.tflib.ops.conv2d
import pkg.tflib.ops.batchnorm
import pkg.tflib.ops.deconv2d
import pkg.tflib.save_images
import pkg.tflib.mnist
import pkg.tflib.plot



def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


def Generator(n_samples, noise=None, useBNdisc=True, layer_dim=64, out_dim=784):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*4*layer_dim, noise)
    if useBNdisc:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4, 4, 4*layer_dim])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*layer_dim, 2*layer_dim, 5, output)
    if useBNdisc:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,1,2], output)
    output = tf.nn.relu(output)

    output = output[:, :7, :7, :]

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*layer_dim, layer_dim, 5, output)
    if useBNdisc:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,1,2], output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', layer_dim, 1, 5, output)
    output = tf.nn.sigmoid(output)
    return tf.reshape(output, [-1, out_dim])


def Discriminator(inputs, keep_prob, name='Discriminator', useBias=True, useBNdisc=False, layer_dim=64):
    name_fixed = 'Discriminator'
    output = tf.reshape(inputs, [-1, 28, 28, 1])

    output = lib.ops.conv2d.Conv2D(name_fixed+'.1', 1, layer_dim, 5, output, stride=2)
    if useBNdisc:
        output = lib.ops.batchnorm.Batchnorm(name_fixed+'.BN1', [0, 1, 2], output)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob)

    output = lib.ops.conv2d.Conv2D(name_fixed+'.2', layer_dim, 2*layer_dim, 5, output, stride=2)
    if useBNdisc:
        output = lib.ops.batchnorm.Batchnorm(name_fixed+'.BN2', [0, 1, 2], output)
    output = LeakyReLU(output)
    output = tf.nn.dropout(output, keep_prob)

    output = lib.ops.conv2d.Conv2D(name_fixed+'.3', 2*layer_dim, 4*layer_dim, 5, output, stride=2)
    if useBNdisc:
        output = lib.ops.batchnorm.Batchnorm(name_fixed+'.BN3', [0, 1, 2], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*layer_dim])
    h = output

    output = lib.ops.linear.Linear(name+'.Output', 4*4*4*layer_dim, 1, output, biases = useBias)
    return tf.reshape(output, [-1]), h
