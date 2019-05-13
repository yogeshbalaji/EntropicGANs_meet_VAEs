import os
import tensorflow as tf
from models import Generator, Discriminator
import pkg.tflib as lib
import gan_utils
import numpy as np
import time


class SWGAN:
    def __init__(self, flags):

        # Parameters
        self.flags = flags
        self.gan_mode = flags.gan_mode
        self.dim = 64  # Model dimensionality
        self.dim_disc = self.dim
        if flags.useDualDisc:
            self.dim_disc = self.dim / 2

        self.batchSize = flags.batch_size  # Batch size
        self.critic_iters = flags.critic_iters
        self.iters = flags.num_iters  # How many generator iterations to train for
        self.output_dim = 784  # Number of pixels in MNIST (28*28)
        self.z_dim = 128

        if flags.useWarmUp:
            print("using warm up objective")

        # Placeholders
        self.real_data = tf.placeholder(tf.float32, shape=[self.batchSize, self.output_dim])
        self.noise_z = tf.random_normal([self.batchSize, self.z_dim])
        self.lamb = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

        self.get_model_params()
        self.form_GAN_cost()

        # Forming save dir
        savedir_split = self.flags.savedir.split('/')
        path_cur = ''
        for d in savedir_split:
            path_cur = os.path.join(path_cur, d)
            if not tf.gfile.Exists(path_cur):
                tf.gfile.MkDir(path_cur)

        self.img_save_file = os.path.join(self.flags.savedir, 'imgs')
        if not tf.gfile.Exists(self.img_save_file):
            tf.gfile.MkDir(self.img_save_file)
        
        if flags.dataset == 'mnist1':
            self.classes_to_load = [1]
        else:
            self.classes_to_load = None

    def get_model_params(self):

        self.fake_data = Generator(self.batchSize, self.noise_z)

        if self.flags.useDualDisc:
            print("use dual disc!")
            self.disc_real, self.real_h = Discriminator(self.real_data, self.keep_prob, name='Discriminator2', useBias=False)
            self.disc_fake, self.fake_h = Discriminator(self.fake_data, self.keep_prob)
            self.gen_params = lib.params_with_name('Generator')
            self.disc_params = lib.params_with_name('Discriminator') + lib.params_with_name('Discriminator2')
        else:
            self.disc_real, self.real_h = Discriminator(self.real_data, self.keep_prob)
            self.disc_fake, self.fake_h = Discriminator(self.fake_data, self.keep_prob)
            self.gen_params = lib.params_with_name('Generator')
            self.disc_params = lib.params_with_name('Discriminator')

        print(self.disc_params)
        print(self.gen_params)

        # For saving samples
        self.saver = tf.train.Saver(max_to_keep=100)
        self.fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
        self.fixed_noise_samples = Generator(128, noise=self.fixed_noise)

    def form_GAN_cost(self):

        if self.gan_mode == 'wgan':
            gen_cost = -tf.reduce_mean(self.disc_fake)
            neg_emd = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
            disc_cost = neg_emd

            self.gen_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
                ).minimize(gen_cost, var_list=self.gen_params)
            self.disc_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5
                ).minimize(disc_cost, var_list=self.disc_params)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(
                    tf.assign(
                        var,
                        tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                    )
                )
            self.clip_disc_weights = tf.group(*clip_ops)

        elif self.gan_mode == 'wgan-gp':
            gen_cost = -tf.reduce_mean(self.disc_fake)
            neg_emd = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
            disc_cost = neg_emd

            alpha = tf.random_uniform(
                shape=[self.batchSize, 1],
                minval=0.,
                maxval=1.
            )
            differences = self.fake_data - self.real_data
            interpolates = self.real_data + (alpha * differences)
            gradients = tf.gradients(Discriminator(interpolates, self.keep_prob), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            disc_cost += self.lamb * gradient_penalty

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4,
                beta1=0.5,
                beta2=0.9
            ).minimize(gen_cost, var_list=self.gen_params)
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4,
                beta1=0.5,
                beta2=0.9
            ).minimize(disc_cost, var_list=self.disc_params)
            self.clip_disc_weights = None

        elif self.gan_mode == 'swgan':
            neg_emd = tf.reduce_mean(self.disc_fake) - tf.reduce_mean(self.disc_real)
            disc_cost = neg_emd
            dist_c_values = gan_utils.dist_c(self.real_data, self.fake_data, useCosineDist=self.flags.useCosineDist)

            approx_assignment = tf.squeeze(tf.expand_dims(self.disc_real, 0)
                                           - tf.expand_dims(self.disc_fake, 1)) - dist_c_values

            smooth_term = gan_utils.strong_convex_func(approx_assignment, lamb=self.lamb,
                                                       useHingedL2=self.flags.useHingedL2)
            disc_cost += smooth_term

            self.sum_probs = gan_utils.sum_probs_func(approx_assignment, lamb=self.lamb)

            self.soft_assignment_mat = gan_utils.strong_convex_func_normalized(approx_assignment, lamb=self.lamb,
                                                                          reduce_mean=False)
            gen_cost_per_examples = tf.stop_gradient(self.soft_assignment_mat) * dist_c_values

            if self.flags.usePrimalLoss:
                gen_cost = tf.reduce_mean(gen_cost_per_examples)
            elif self.flags.useDualLoss:
                gen_cost = -disc_cost
            else:
                gen_cost = -tf.reduce_mean(self.disc_fake)

            gen_grads = tf.gradients(gen_cost, self.gen_params)
            disc_grads = tf.gradients(disc_cost, self.disc_params)

            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=self.flags.lr_gen,
                beta1=0.5,
                beta2=0.99, epsilon=1e-8,
            ).apply_gradients(zip(gen_grads, self.gen_params))
            self.disc_train_op = tf.train.AdamOptimizer(
                learning_rate=self.flags.lr_disc,
                beta1=0.5,
                beta2=0.99, epsilon=1e-8,
            ).apply_gradients(zip(disc_grads, self.disc_params))

            self.clip_disc_weights = None
            self.smooth_term = smooth_term
            self.dist_c_values = dist_c_values

        self.disc_cost = disc_cost
        self.gen_cost = gen_cost
        self.neg_emd = neg_emd

    def generate_image(self, frame, session):
        samples = session.run(self.fixed_noise_samples)
        lib.save_images.save_images(
            samples.reshape((128, 28, 28)),
            '{}/samples_{}_{}.png'.format(self.img_save_file, self.flags.savename, frame)
        )

    def train(self):

        # Dataset iterator
        train_gen, dev_gen, test_gen = lib.mnist.load(self.flags.batch_size, self.flags.batch_size, classes_to_load=self.classes_to_load)
        save_iter_list = [1, 100, 500, 1000, 5000, 10000, 49000]

        # Train loop
        with tf.Session() as session:

            session.run(tf.global_variables_initializer())
            gen = gan_utils.inf_train_gen(train_gen)

            for iteration in range(self.iters):
                start_time = time.time()
                disc_iters = self.critic_iters

                _data = gen.next()
                _noise_z = np.random.randn(self.batchSize, 128)

                if self.gan_mode == 'swgan':
                    if iteration < 49:
                        _LAMBDA = self.flags.Lambda * 2.
                    elif iteration < 99:
                        _LAMBDA = self.flags.Lambda * 2. * 0.8
                    elif iteration < 199:
                        _LAMBDA = self.flags.Lambda * 2. * 0.8 * 0.8
                    elif iteration < 299:
                        _LAMBDA = self.flags.Lambda * 1.
                    else:
                        if self.flags.adaptLambda:
                            print("Adapting lambda")
                            if _neg_emd > 0.:
                                if _sum_probs < 1.3 and _sum_probs > 0.8:
                                    _LAMBDA /= 0.9
                            elif np.abs(_smooth_term) / np.abs(_neg_emd) > 0.2:
                                if _sum_probs < 1.3 and _sum_probs > 0.8:
                                    _LAMBDA *= 0.9
                else:
                    _LAMBDA = self.flags.Lambda

                # Training discriminator
                for i in range(disc_iters):
                    if self.gan_mode == 'swgan':
                        _data = gen.next()
                        
                        _noise_z = np.random.randn(self.batchSize, 128)
                        disc_train = self.disc_train_op
                        _disc_cost, _neg_emd, _smooth_term, _sum_probs, _ = session.run(
                            [self.disc_cost, self.neg_emd, self.smooth_term, self.sum_probs, disc_train],
                            feed_dict={self.real_data: _data, self.keep_prob: 1., self.noise_z: _noise_z,
                                       self.lamb: _LAMBDA}
                        )
                    else:
                        _data = gen.next()
                        _disc_cost, _neg_emd, _ = session.run(
                            [self.disc_cost, self.neg_emd, self.disc_train_op],
                            feed_dict={self.real_data: _data, self.keep_prob: 1., self.lamb: _LAMBDA}
                        )
                    if self.clip_disc_weights is not None:
                        _ = session.run(self.clip_disc_weights)

                if iteration % 50 == 0:
                    print("iteration: {}".format(iteration))
                    print("sum_prob for disc: ", _sum_probs)
                    print("smooth term for disc: ", _smooth_term)
                    print("smooth term (without lambda) for disc: ", _smooth_term * _LAMBDA)
                    print("neg EMD: ", _neg_emd)

                # Training generator
                if iteration > 0:
                    if self.gan_mode == 'swgan':
                        gen_train = self.gen_train_op
                        num_gen_updates = 1

                        for j in range(num_gen_updates):
                            if _neg_emd < 0.:
                                if True:
                                    _, _sum_probs, _soft_assignment_mat, _dist_c_values = session.run(
                                        [gen_train, self.sum_probs, self.soft_assignment_mat, self.dist_c_values],
                                        feed_dict={self.real_data: _data, self.keep_prob: 1., self.noise_z: _noise_z,
                                                   self.lamb: _LAMBDA, })
                                    if iteration % 50 == 0:
                                        print("sum_prob for gen: ", _sum_probs)
                                        np.set_printoptions(precision=2)

                                else:
                                    print("skip")
                            else:
                                if iteration % 50 == 0:
                                    print("skip due to poor disc_cost {}".format(_disc_cost))
                            if iteration % 50 == 0:
                                print(_disc_cost)
                    else:
                        _ = session.run(self.gen_train_op, feed_dict={self.real_data: _data, self.keep_prob: 1.,
                                                                      self.lamb: _LAMBDA})

                if iteration % 50 == 0:
                    print("LAMBDA: {}".format(_LAMBDA))
                    print("############")

                if self.gan_mode == 'swgan':
                    lib.plot.plot('train smooth term %s' % (self.flags.savename), _smooth_term)
                lib.plot.plot('train disc cost %s' % (self.flags.savename), _disc_cost)
                lib.plot.plot('train neg EMD %s' % (self.flags.savename), _neg_emd)
                lib.plot.plot('time', time.time() - start_time)

                # Calculate dev loss and generate samples every 100 iters
                if iteration % 500 == 0 or iteration in save_iter_list:
                    dev_disc_costs = []
                    dev_neg_emd_costs = []
                    for images, _ in dev_gen():
                        _dev_disc_cost, _neg_emd = session.run(
                            [self.disc_cost, self.neg_emd],
                            feed_dict={self.real_data: images, self.keep_prob: 1.0, self.lamb: _LAMBDA}
                        )
                        dev_disc_costs.append(_dev_disc_cost)
                        dev_neg_emd_costs.append(_neg_emd)
                    lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))
                    lib.plot.plot('dev neg EMD', np.mean(dev_neg_emd_costs))

                    self.generate_image(iteration, session)

                # Write logs every 100 iters
                if (iteration < 5) or (iteration % 100 == 99):
                    lib.plot.flush()
                lib.plot.tick()

                if iteration % 5000 == 0:
                    if not tf.gfile.Exists(os.path.join(self.flags.savedir, 'models')):
                        tf.gfile.MkDir(os.path.join(self.flags.savedir, 'models'))
                    self.saver.save(session, '%s/models/model_%d.ckpt' % (self.flags.savedir, iteration))

                if iteration in save_iter_list:
                    if not tf.gfile.Exists(os.path.join(self.flags.savedir, 'models_plot')):
                        tf.gfile.MkDir(os.path.join(self.flags.savedir, 'models_plot'))
                    self.saver.save(session, '%s/models_plot/model_%d.ckpt' % (self.flags.savedir, iteration))
