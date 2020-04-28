# -*- coding: utf-8 -*-
"""
Created on March 31 10:08:39 2017
cnv_sim_delfi

@author: grace
"""
import random
import argparse

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import scipy.stats
import sympy
import seaborn as sns
import pandas as pd
import numba
from scipy.stats import entropy
#plt.warnings.filterwarnings('ignore', "The 'normed' kwarg is deprecated")

red, blue, green = sns.color_palette('Set1', 3)

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-cs', "--cnv_fitness_effect")
parser.add_argument('-cu', "--cnv_mutation_rate")
parser.add_argument('-ss', "--snv_fitness_effect")
parser.add_argument('-su', "--snv_mutation_rate")
#parser.add_argument('-t', "--threads")
parser.add_argument('-s', "--seed")
parser.add_argument('-o', "--outplot")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
outfile = str(args.outplot)
true_params = np.log10(np.array([float(args.cnv_fitness_effect), float(args.cnv_mutation_rate)]))  
s_snv = float(args.snv_fitness_effect)
m_snv = float(args.snv_mutation_rate)

# threads
#n_processes = int(args.threads)

#### Prior over model parameters ####
import delfi.distribution as dd

seed_p = 2
prior_min = np.log10(np.array([1e-3,1e-12]))
prior_max = np.log10(np.array([0.3,1e-4]))
prior = dd.Uniform(lower=prior_min, upper=prior_max,seed=seed_p)

#### simulation ####
def CNVsimulator(N, s_snv, m_snv, cnv_params, generation, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for 267 generations
    Returns proportion of the population with a CNV per generation as 1d np.array of length 267
    
    Parameters
    -------------------
    N : int
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    cnv_params : np.array, 1d of length dim_param
            Parameter vector with nv selection coefficient and cnv mutation rate
    gen : np.array, 1d 
        with generations 0 through the end of simulation
    seed : int
    """
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    assert N > 0
    N = np.uint64(N)
    s_cnv, m_cnv = cnv_params
    
    w = np.array([1, 1 + s_cnv, 1 + s_snv])
    S = np.diag(w)
    
    # make transition rate array
    # make transition rate array
    M = np.array([[1 - m_cnv - m_snv, 0, 0],
                [m_cnv, 1, 0],
                [m_snv, 0, 1]])
    assert np.allclose(M.sum(axis=0), 1)
    
    # mutation and selection
    E = M @ S

    # rows are genotypes
    n = np.zeros(3)
    n[0] = N  
    
    # follow proportion of the population with CNV
    # here rows with be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation 267
    for t in generation:    
        p = n/N  # counts to frequencies
        p_cnv.append(p[1,0])  # frequency of CNVs
        p = E @ p.reshape((3, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, p) # random genetic drift

    return np.transpose(p_cnv)

from delfi.simulator.BaseSimulator import BaseSimulator

class CNVevo(BaseSimulator):
    def __init__(self, N, s_snv, m_snv, generation, seed=None):
        """ CNV evolution simulator
        Simulates CNV and SNV evolution for 267 generations
        Returns proportion of the population with a CNV per generation as 1d np.array of length 267
    
        Parameters
        -------------------
        N : int
            population size  
        s_snv : float
            fitness benefit of SNVs  
        m_snv : float 
            probability mutation to SNV
        gen : np.array, 1d 
            with generations 0 through the end of simulation
        seed : int or None
            If set, randomness across runs is disabled
        """
        dim_param = 2

        super().__init__(dim_param=dim_param, seed=seed)
        self.N = N
        self.s_snv = s_snv
        self.m_snv = m_snv
        self.generation = generation
        self.CNVsimulator = CNVsimulator

    def gen_single(self, cnv_params):
        """Forward model for simulator for single parameter set

        Parameters
        ----------
        params : list or np.array, 1d of length dim_param
            Parameter vector

        Returns
        -------
        dict : dictionary with data
            The dictionary must contain a key data that contains the results of
            the forward run. Additional entries can be present.
        """
        params = np.asarray(cnv_params)

        assert params.ndim == 1, 'params.ndim must be 1'

        sim_seed = self.gen_newseed()
        states = self.CNVsimulator(N, s_snv, m_snv, cnv_params, generation, seed=sim_seed)
        
        return {'data': states.reshape(-1),
                'generation': self.generation,
                's_snv': self.s_snv,
                'm_snv': self.m_snv,
                'N': self.N}

#### summary stats ####
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats
from scipy import stats as spstats

class CNVStats(BaseSummaryStats):
    """Moment based SummaryStats class for the CNV evolution model

    Calculates summary statistics
    """
    def __init__(self, n_summary=25, seed=None):
        """See SummaryStats.py for docstring"""
        super(CNVStats, self).__init__(seed=seed)
        self.n_summary = n_summary

    def calc(self, repetition_list):
        """Calculate summary statistics

        Parameters
        ----------
        repetition_list : list of dictionaries, one per repetition
            data list, returned by `gen` method of Simulator instance

        Returns
        -------
        np.array, 2d with n_reps x n_summary
        """
        exp_gen = np.array([25,33,41,54,62,70,79,87,95,103,116,124,132,145,153,161,174,182,190,211,219,232,244,257,267])
        stats = []
        for r in range(len(repetition_list)):
            #x = repetition_list[r]
            cnv_freq = np.transpose(repetition_list[r]['data'])
            subset = cnv_freq[exp_gen]
            stats.append(subset)

        return np.asarray(stats)

#### generator ####
import delfi.generator as dg

# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3e8
generation = np.array(range(0,268))

# summary statistics hyperparameters
n_summary = 25

## for local ##
seed_m = argseed
m = CNVevo(N, s_snv, m_snv, generation, seed=seed_m)
s = CNVStats(n_summary = n_summary)
g = dg.Default(model=m, prior=prior, summary=s)

#### true params and data ####
# true parameters and respective labels
# true parameters obtained from command line     
labels_params = ['CNV fitness effect', 'CNV mutation rate']
obs = m.gen_single(true_params) # one observed pop
#single observation summary stats
obs_stats = s.calc([obs])

#### inference ####
## hyperparameters ##
seed_inf = 1
pilot_samples = 2000
# training schedule
n_train = 2000
n_rounds = 1
# fitting setup
minibatch = 256
epochs = 100
val_frac = 0.05
# network setup
n_hiddens = [50,50]
# convenience
prior_norm = True
# MAF parameters
density = 'maf'
n_mades = 5

## inference ##
import delfi.inference as infer

# inference object
res = infer.APT(g,
                  obs=obs_stats,
                  n_hiddens=n_hiddens,
                  seed=seed_inf,
                  pilot_samples=pilot_samples,
                  n_mades=n_mades,
                  prior_norm=prior_norm,
                  density=density)
# train
log, _, posterior = res.run(
                    n_train=n_train,
                    n_rounds=n_rounds,
                    minibatch=minibatch,
                    epochs=epochs,
                    silent_fail=False,
                    proposal='prior',
                    val_frac=val_frac,
                    verbose=True,)

#### parse posterior ####
posterior_samples = [posterior[0].gen(10000)]
fitness_samples = posterior_samples[0][:,0]
mut_samples = posterior_samples[0][:,1]

#calculations kde
ymin = fitness_samples.min()
ymax = fitness_samples.max()
xmin = mut_samples.min()
xmax = mut_samples.max()

s_range, μ_range = np.mgrid[ymin:ymax:100j, xmin:xmax:100j]
positions = np.vstack([s_range.ravel(), μ_range.ravel()])
values = np.vstack([fitness_samples, mut_samples])
kernel = scipy.stats.gaussian_kde(values)
density = np.reshape(kernel(positions).T, s_range.shape)

#estimates for parameters from the posterior (MAP - highest probability in posterior)
idx = np.argmax(density, axis=None)
s_est = positions[0,idx]
μ_est = positions[1,idx]

#### plots to output to pdf ####
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

# observed
axes[0,0].plot(obs['generation'],obs['data'])
axes[0,0].set(ylabel='CNV frequency', xlabel='Generation')
axes[0,0].set_title('observed data\nfitness: ' + str(true_params[0]) + '\nmutation: ' + str(true_params[1]))

axes[0,1].plot(log[0]['loss'],lw=2)
axes[0,1].set(xlabel='iteration', ylabel='loss')

# marginal
sns.distplot(fitness_samples, bins=50, fit=scipy.stats.norm, kde=False, ax=axes[1,0]) #change the fit
sns.distplot(mut_samples, bins=50, fit=scipy.stats.norm, kde=False, ax=axes[2,1]) # change the fit

axes[1,0].axvline(true_params[0], color=red)
axes[1,0].axvline(s_est, color=green)
axes[2,1].axvline(true_params[1], color=red)
axes[2,1].axvline(μ_est, color=green)


# joint
axes[1,1].pcolormesh(μ_range, s_range, density)
axes[1,1].plot(true_params[1], true_params[0], color=red, marker='o')
axes[1,1].plot(μ_est, s_est, color=green, marker='o')

axes[2,0].plot(mut_samples, fitness_samples, ',k')
axes[2,0].plot(true_params[1], true_params[0], color=red, marker='o')
axes[2,0].plot(μ_est, s_est, color=green, marker='o')
axes[2,0].contour(μ_range, s_range, density, colors='k', linewidths=1)

axes[1,0].set(xlabel='log10(CNV fitness effect)')
axes[1,1].set(xlabel='log10(CNV mutation rate)', ylabel='log10(CNV fitness effect)')
axes[2,0].set(xlabel='log10(CNV mutation rate)', ylabel='log10(CNV fitness effect)')
axes[2,1].set(xlabel='log10(CNV mutation rate)')


fig.tight_layout()
plt.title('')
sns.despine()
plt.savefig(outfile + '.pdf')  

#### Write out the estimated parameters and "true" parameters ####
# single file for all combos
def format(value):
    return "%.12f" % value

f= open("est_real_params.csv","a+")
f.write(','.join(str(format(j)) for j in (true_params[0],s_est,true_params[1],μ_est,s_snv,m_snv)) + '\n')
f.close() 
