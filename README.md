# EntropicGANs_meet_VAEs
Official implementation of "EntropicGANs meet VAEs: A Statistical Approach to Compute Sample Likelihoods in GANs"

![Alt text](figs/framework.pdf?raw=true "Framework")

To train the model, run
```
python src/main.py --savename mnist --savedir results/MNIST --gan_mode swgan --usePrimalLoss
```

To compute sample likelihoods on the trained EntropicGAN model, run
```
python src/main.py --savename mnist --savedir results/MNIST --loadpath results/MNIST/models/model_5000.ckpt --gan_mode swgan --mode eval --evalroot 'path to dataset whose likelihood we wish to compute'
```

The likelihood scores for samples are stored as a numpy array.

