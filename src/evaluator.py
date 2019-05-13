import os
import tensorflow as tf
from models import Generator, Discriminator
import pkg.tflib as lib
import gan_utils
import numpy as np
import time
import dataloader
import gan_utils


class LikelihoodComputation:
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
        self.output_dim = 784  # Number of pixels in MNIST (28*28)
        self.z_dim = 128
        self.lamb = 5.0

        if flags.useWarmUp:
            print("using warm up objective")

        # Placeholders
        self.real_data = tf.placeholder(tf.float32, shape=[1, self.output_dim])
        self.noise_z = tf.random_normal([self.batchSize, self.z_dim])
        self.keep_prob = tf.placeholder(tf.float32)
        self.fake_data_gen = Generator(self.batchSize, self.noise_z)
        self.fake_data = tf.placeholder(tf.float32, shape=[self.batchSize, self.output_dim])

        D1_y, __ = Discriminator(self.real_data, self.keep_prob, name='Discriminator2', useBias=False)
        D2_yhat, __ = Discriminator(self.fake_data, self.keep_prob)
        dist_c_values = tf.squeeze(gan_utils.dist_c(self.real_data, self.fake_data, useCosineDist=flags.useCosineDist))
        self.v_compute = D1_y - D2_yhat - dist_c_values
        self.saver = tf.train.Saver()


        # Forming save dir
        savedir_split = self.flags.savedir.split('/')
        path_cur = ''
        for d in savedir_split:
            path_cur = os.path.join(path_cur, d)
            gan_utils.mkdirp(path_cur)

        if flags.dataset == 'mnist1':
            self.classes_to_load = [1]
        else:
            self.classes_to_load = None


    def numpy_dist_c(self, x, y):
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 1)
        dist = np.sum(np.abs(x - y), 2)
        return dist

    def v_star(self, y, y_hat, session):
        
        nbatches = int(y_hat.shape[0]/self.batchSize)
        v_val_all = []
        for i in range(nbatches):
            y_hat_batch = y_hat[i*self.batchSize: (i+1)*self.batchSize, :]
            v_val = session.run(self.v_compute, feed_dict={self.real_data:y, self.fake_data: y_hat_batch, self.keep_prob:1.})
            v_val_all.append(v_val)
        v_val_all = np.vstack(v_val_all)
        v_val_all = np.reshape(v_val_all, -1)
        
        return v_val_all
        
    def generate_samples(self, nsamples, session):
        # Function to generate images from noise vectors
        
        nbatches = int(nsamples/self.batchSize)
        gen_imgs_all = []
        noise_all = []
        for i in range(nbatches):
            noise_batch = np.random.randn(self.batchSize, self.z_dim)
            noise_all.append(noise_batch)
            
            [gen_imgs] = session.run(
                      [self.fake_data_gen],
                      feed_dict={self.noise_z:noise_batch})
            gen_imgs = gen_imgs.reshape((self.batchSize, 28, 28))
            gen_imgs_all.append(gen_imgs)
            
        gen_imgs_all = np.vstack(gen_imgs_all)
        noise_all = np.vstack(noise_all)
        return noise_all, gen_imgs_all        

    def estimate_probs(self, x, y_hat, y, session):
        
        # unnormalized density
        term1 = np.exp(np.diag(-0.5*np.dot(x, x.T)))
        v_star_val = self.v_star(y, y_hat, session)/self.lamb
        # to avoid numerical instability
        v_star_val = v_star_val - np.max(v_star_val)
        term2 = np.exp(v_star_val)
        prob_vals = term1*term2

        #normlizing the density
        prob_vals = prob_vals/np.sum(prob_vals)
        return prob_vals

    def compute_LL(self, y, session, save_imgs=False, item_num=-1):
        # Function to compute log-likelihood of a new sample y
        nsamples = self.batchSize*5
        eps = 10**-70
        
        x, y_hat = self.generate_samples(nsamples, session)
        y_hat = np.reshape(y_hat, (y_hat.shape[0], self.output_dim))
        prob = self.estimate_probs(x, y_hat, y, session)
        
        c = np.squeeze(self.numpy_dist_c(y_hat, y))
        term1 = np.sum(c*prob)
        
        # Debugging by visualization
        if save_imgs == True:
            ll_path = os.path.join(self.flags.savedir, 'll_imgs_debug')
            gan_utils.mkdirp(ll_path)
            save_path_orig = os.path.join(ll_path, 'orig')
            save_path_gen = os.path.join(ll_path, 'gen_samples')
            gan_utils.mkdirp(save_path_orig)
            gan_utils.mkdirp(save_path_gen)
            
            lib.save_images.save_images(
                y.reshape((y.shape[0], 28, 28)),
                '{}/samples_{}.png'.format(save_path_orig, item_num)
            )
            lib.save_images.save_images(
                y_hat.reshape((y_hat.shape[0], 28, 28)),
                '{}/samples_{}.png'.format(save_path_gen, item_num)
            )
            
        # computing entropy
        ent = np.sum(-prob*np.log(prob+eps))
        
        # computing likelihood of the latent variable
        term3 = np.sum((-0.5*np.sum(x**2, 1))*prob)
        
        upper_bound = (-1/self.lamb)*(term1 - self.lamb*ent) + term3
        return upper_bound    
    
    def compute_LL_dataset(self):
        
        eval_root = self.flags.evalroot
        load_path = self.flags.loadpath
        
        if self.flags.eval_savepath == '':
            save_path = os.path.join(self.flags.savedir, 'LL')
            gan_utils.mkdirp(save_path)
            save_path = os.path.join(save_path, self.flags.dataset)
            gan_utils.mkdirp(save_path)
        else:
            save_path = self.flags.eval_savepath
            save_path_split = save_path.split('/')
            cur_path = ''
            for pth in save_path_split:
                cur_path = os.path.join(cur_path, pth)
                gan_utils.mkdirp(cur_path)
            
        nsamples_eval = self.flags.nsamples_eval
        
        with tf.Session() as session:
            
            session.run(tf.global_variables_initializer())
            self.saver.restore(session, load_path)
            print('Model restored successfully from {}'.format(load_path))
            
            test_data, filenames = dataloader.read_test_data(eval_root, nsamples_eval, classes_to_load=self.classes_to_load, return_filenames=True)
            LL_list = []
            
            for i in range(len(test_data)):
                if i%50 == 0:
                    print('{}/{} samples evaluated'.format(i, len(test_data))   )
                # For each sample, let us estimate Px|y using finite number of samples
                LL = self.compute_LL(test_data[i], session, save_imgs=False, item_num=i)
                LL_list.append(LL)
            
            LL_list = np.array(LL_list)
            LL_sorted = np.argsort(LL_list)
            index_min5 = LL_sorted[0:5]
            index_max5 = LL_sorted[-6:-1]
            
            fil = open('{}/LL_stats.txt'.format(save_path), 'w')
            lines = ['Top 5 min LL files \n']
            for index_min in index_min5:
                line = filenames[index_min] + '\n'
                lines.append(line)

            lines.append('Top 5 max LL files \n')
            for index_max in index_max5:
                line = filenames[index_max] + '\n'
                lines.append(line)

            fil.writelines(lines)
            fil.close()
            
            print(LL_list)
            
            np.save('{}/LL.npy'.format(save_path), LL_list)
            print('LL computation done')
        
