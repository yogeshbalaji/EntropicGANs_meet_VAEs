# Code for generating plots

import numpy as np 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.mlab as mlab	
import os
import seaborn as sns
import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_root', required=True, help='path to  LL files')
    parser.add_argument('--savedir', default='results/MNIST1', help='path to save plots')
    args = parser.parse_args()
    
    ll_npy_list = []
    dataset_list = []
    for fol in os.listdir(args.ll_root):
        ll_npy_list.append(os.path.join(args.ll_root, fol, 'LL.npy'))
        dataset_list.append(fol.upper())
    
    # color_vals = ['y', 'g', 'r', 'b']
    color_vals = ['r', 'g', 'b', 'y'] 
    
    bin_width = 1.2
    
    for i, ll_npy_file in enumerate(ll_npy_list):
        LL_vals = np.load(ll_npy_file)    
        nbins = int((max(LL_vals) - min(LL_vals))/bin_width)
        weights = np.ones_like(LL_vals)/len(LL_vals)
        n, bins, patches = plt.hist(LL_vals, nbins, color=color_vals[i], alpha=0.3, weights=weights)
        sns.kdeplot(LL_vals, color=color_vals[i])

    plt.xlabel(r'$log(f_{Y}(y^{test}))$')
    plt.ylabel('Density')
    plt.legend(tuple(dataset_list))
    plt.savefig('{}/LL_plot.png'.format(args.savedir), bbox_inches='tight', format='png', dpi=800)


if __name__ == '__main__':
    main()
