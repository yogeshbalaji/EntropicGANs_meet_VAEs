import tensorflow as tf


def dist_c(x, y, diag=False, useCosineDist=False):
    if useCosineDist:
        print('using cosine dist_c')
        if diag:
            norms = tf.sqrt(tf.reduce_sum(y ** 2, 1)) * tf.sqrt(tf.reduce_sum(x ** 2, 1))
            return (1. - tf.reduce_sum(y * x, 1) / (norms + 1e-5)) * 100.
        else:
            norms = tf.matmul(tf.sqrt(tf.reduce_sum(y ** 2, 1, keep_dims=True)),
                              tf.sqrt(tf.reduce_sum(x ** 2, 1, keep_dims=True)), transpose_b=True)
            return (1. - tf.matmul(y, x, transpose_b=True) / (norms + 1e-5)) * 100.
    else:
        print('using l1 dist_c')
        if diag:
            return tf.reduce_mean(tf.abs(x - y), 1)
        else:
            x = tf.expand_dims(x, 0)
            y = tf.expand_dims(y, 1)
            return tf.reduce_sum(tf.abs(x - y), 2)


def strong_convex_func(x, lamb, reduce_mean=True, useHingedL2=False):
    if useHingedL2:
        func = (tf.maximum(x, 0) ** 2) / lamb / 2.
    else:
        func = tf.exp(x / lamb) / tf.exp(1.) * lamb

    if reduce_mean:
        return tf.reduce_mean(func)
    else:
        return func


def strong_convex_func_normalized(x, lamb, reduce_mean=False, useHingedL2=False):
    if useHingedL2:
        func = (tf.maximum(x, 0) ** 2) / 2.
    else:
        func = tf.exp(x / lamb) / tf.exp(1.)

    if reduce_mean:
        return tf.reduce_mean(func)
    else:
        return func


def sum_probs_func(x, lamb):
    return tf.reduce_mean(tf.maximum(x, 0.0)) / lamb


def inf_train_gen(train_gen):
    while True:
        for images, targets in train_gen():
            yield images


def mkdirp(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MkDir(path)

