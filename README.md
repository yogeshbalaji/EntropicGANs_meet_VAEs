# EntropicGANs_meet_VAEs
Official implementation of "EntropicGANs meet VAEs: A Statistical Approach to Compute Sample Likelihoods in GANs"

<p align="center">
  <img src="figs/framework.png" width="500">
</p>

## Training 
To train the model, run
```
python src/main.py --savename mnist --savedir results/MNIST --gan_mode swgan --usePrimalLoss
```

To compute sample likelihoods on the trained EntropicGAN model, run
```
python src/main.py --savename mnist --savedir results/MNIST --loadpath results/MNIST/models/model_5000.ckpt --gan_mode swgan --mode eval --evalroot 'path to dataset whose likelihood we wish to compute'
```

The likelihood scores for samples are stored as a numpy array.

<p align="center">
  <img src="figs/LL_iterations_ICML.png" width="500">
</p>

<p align="center">
  <img src="figs/LL_datasets_ICML.png" width="500">
</p>

## Citation

If you use this code for your research, please cite

    @article{Balaji2018Entropic,
    author    = {Yogesh Balaji and
                 Hamed Hassani and
                 Rama Chellappa and
                 Soheil Feizi},
    title     = {Entropic GANs meet VAEs: {A} Statistical Approach to Compute Sample
                 Likelihoods in GANs},
    journal   = {CoRR},
    volume    = {abs/1810.04147},
    year      = {2018},
    url       = {http://arxiv.org/abs/1810.04147},
    }

