import os, sys
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import pkg.tflib as lib
import trainer
import evaluator


def main():
    flags = tf.flags
    flags.DEFINE_string('savename', '',"""generated png image name""")
    flags.DEFINE_string('savedir', 'results/MNIST1',"""save directory""")
    flags.DEFINE_string('gan_mode', 'swgan',"""swgan, wgan, or wgan-gp""")
    flags.DEFINE_string('dataset', 'mnist',"""mnist, mnist1, svhn""")
    flags.DEFINE_float('lr_gen', 2e-4,"""lr_gen""")
    flags.DEFINE_float('mom_gen', 0.5,"""momentum""")
    flags.DEFINE_float('lr_disc', 2e-4,"""lr_disc""")
    flags.DEFINE_float('mom_disc', 0.5,"""momentum""")
    flags.DEFINE_integer('critic_iters', 5, "Number of steps for the discriminator per step of the generator")
    flags.DEFINE_float('Lambda', 5.,"""LAMBDA""")
    flags.DEFINE_integer('batch_size', 128, "batch size")
    flags.DEFINE_integer('num_iters', 10000, "batch size")
    flags.DEFINE_boolean('useHingedL2', False, "use hinged L2 or entropy")
    flags.DEFINE_boolean('useCosineDist', False, "use cosine or L1")
    flags.DEFINE_boolean('usePrimalLoss', False, "optimize pi hat")
    flags.DEFINE_boolean('useDualLoss', True, "optimize the entire loss func")
    flags.DEFINE_boolean('useDualDisc', True, "use 2 Disc")
    flags.DEFINE_boolean('useBNdisc', True, "use BN on Disc")
    flags.DEFINE_boolean('useWarmUp', True, "use adaptive warmup")
    flags.DEFINE_boolean('adaptLambda', False, "use adaptive lambda")
    
    flags.DEFINE_string('mode', 'train',"""Train - train entropicGAN, eval - Compute likelihoods""")
    
    # Eval params
    flags.DEFINE_string('evalroot', 'mnist',"""Path to the eval dataset""")
    flags.DEFINE_string('loadpath', 'results/MNIST1/models/model_5000.ckpt',"""Path to trained model""")
    flags.DEFINE_integer('nsamples_eval', 1000, "Number of samples to compute the likelihood for")
    flags.DEFINE_string('eval_savepath', '',"""Path to save evaluation LL scores""")
    
    FLAGS = flags.FLAGS

    lib.print_model_settings(locals().copy())

    if FLAGS.mode == 'train':
        _trainer = trainer.SWGAN(FLAGS)
        _trainer.train()
    elif FLAGS.mode == 'eval':
        _evaluator = evaluator.LikelihoodComputation(FLAGS)
        _evaluator.compute_LL_dataset()


if __name__ == '__main__':
    main()
