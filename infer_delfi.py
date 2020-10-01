# -*- coding: utf-8 -*-
"""
Created on March 31 10:08:39 2020
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
from cnv_simulation import CNVsimulator_simpleWF, CNVsimulator_simpleChemo

red, blue, green = sns.color_palette('Set1', 3)

#### arguments ####
# cnv_sim_delfi.py -m "WF" -cs 0.001 -cu 1e-7 -ss 0.001 -su 1e-5 -t 3 -s $SLURM_ARRAY_TASK_ID -o "snvWSSM_cnvWSWM$SLURM_ARRAY_TASK_ID"
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-cs', "--cnv_fitness_effect")
parser.add_argument('-cu', "--cnv_mutation_rate")
parser.add_argument('-ss', "--snv_fitness_effect")
parser.add_argument('-su', "--snv_mutation_rate")
parser.add_argument('-t', "--threads")
parser.add_argument('-s', "--seed")
parser.add_argument('-o', "--outplot")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
outfile = str(args.outplot)
EvoModel = str(args.model)
true_params = np.log10(np.array([float(args.cnv_fitness_effect), float(args.cnv_mutation_rate)]))  
s_snv = float(args.snv_fitness_effect)
m_snv = float(args.snv_mutation_rate)

# threads
n_processes = int(args.threads)

#### Prior over model parameters ####
import delfi.distribution as dd

seed_p = 2
prior_min = np.log10(np.array([1e-4,1e-12]))
prior_max = np.log10(np.array([0.3,1e-3]))
prior = dd.Uniform(lower=prior_min, upper=prior_max,seed=seed_p)

## simulation ##
from delfi.simulator.BaseSimulator import BaseSimulator

class CNVevo(BaseSimulator):
    def __init__(self, N, s_snv, m_snv, generation, EvoModel, seed=None):
        """ CNV evolution simulator
        Simulates CNV and SNV evolution for 267 generations
        Returns proportion of the population with a CNV per generation as 1d np.array of length 25
    
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
        EvoModel : str
            whether to use WF or chemostat
        seed : int or None
            If set, randomness across runs is disabled
        """
        dim_param = 2

        super().__init__(dim_param=dim_param, seed=seed)
        self.N = N
        self.s_snv = s_snv
        self.m_snv = m_snv
        self.generation = generation
        self.CNVsimulator_simpleWF = CNVsimulator_simpleWF
        self.CNVsimulator_simpleChemo = CNVsimulator_simpleChemo
        self.EvoModel = EvoModel
        

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
        
        if self.EvoModel == "WF":
            states = self.CNVsimulator_simpleWF(N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, seed=sim_seed, cnv_params=cnv_params)
        if self.EvoModel == "Chemo":
            states = self.CNVsimulator_simpleChemo(A_inoc=A_inoc, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, I=I, y=y, τ=τ, seed=sim_seed, cnv_params=cnv_params)
        
        return {'data': states.reshape(-1),
                'generation': np.array([25,33,41,54,62,70,79,87,95,103,116,124,132,145,153,161,174,182,190,211,219,232,244,257,267]),
                's_snv': self.s_snv,
                'm_snv': self.m_snv,
                'N': self.N}
    
#### summary stats ####
from delfi.summarystats.BaseSummaryStats import BaseSummaryStats

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
        stats = []
        for r in range(len(repetition_list)):
            cnv_freq = np.transpose(repetition_list[r]['data'])
            stats.append(cnv_freq)

        return np.asarray(stats)

#### generator ####
import delfi.generator as dg

# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3e8
generation = np.array(range(0,268))

# chemostat parameters
A_inoc = 1e5
S_init = .800
D=0.12
μA=0.45
k=.103
y=3244500
I=.800
τ=1/10

# summary statistics hyperparameters
n_summary = 25

## for HPC ##
seeds_m = np.arange(1,n_processes+1,1)
m = []
s = CNVStats(n_summary = n_summary)
for i in range(n_processes):
    m.append(CNVevo(N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, EvoModel=EvoModel, seed=seeds_m[i]))
g = dg.MPGenerator(models=m, prior=prior, summary=s)

#### true params and data ####
# true parameters and respective labels
# true parameters obtained from command line     
labels_params = ['CNV fitness effect', 'CNV mutation rate']
obs = m[0].gen_single(true_params) # one observed pop
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
param_guess = np.array([positions[0,idx],positions[1,idx]])
def kernelminus(x):
    return -kernel(x)
s_est, μ_est = scipy.optimize.minimize(kernelminus,param_guess, method ='Nelder-Mead', 
 options={'disp': True}).x


#### plots to output to pdf ####
fig, axes = plt.subplots(3, 2, figsize=(10, 10))

# observed
axes[0,0].plot(obs['generation'],obs['data'])
axes[0,0].set(ylabel='Proportion of pop with CNV', xlabel='Generation')
axes[0,0].set_title('observed data\nmodel:'+ EvoModel +'\nlog10(CNV fitness): ' + str(true_params[0]) + '\nlog10(CNV mutation rate): ' + 
                    str(true_params[1]) + '\nSNV fitness:' + str(s_snv) + '\nSNV mutation rate:' + str(m_snv))

axes[0,1].plot(log[0]['loss'],lw=2)
axes[0,1].set(xlabel='iteration', ylabel='loss')

# marginal
sns.distplot(fitness_samples, bins=50, kde=False, ax=axes[1,0]) 
sns.distplot(mut_samples, bins=50, kde=False, ax=axes[2,1]) 

axes[1,0].axvline(true_params[0], color=red, label="simulation parameter")
axes[1,0].axvline(s_est, color=green,label="MAP estimate")
axes[1,0].legend()
axes[2,1].axvline(true_params[1], color=red,label="simulation parameter")
axes[2,1].axvline(μ_est, color=green,label="MAP estimate")
axes[2,1].legend()

prior_min = np.log10(np.array([1e-4,1e-12]))
prior_max = np.log10(np.array([0.3,1e-3]))

s_range, μ_range = np.mgrid[np.log10(1e-4):np.log10(0.3):100j, np.log10(1e-12):np.log10(1e-3):100j]
positions = np.vstack([s_range.ravel(), μ_range.ravel()])
values = np.vstack([fitness_samples, mut_samples])
kernel = scipy.stats.gaussian_kde(values)
density = np.reshape(kernel(positions).T, s_range.shape)

# joint
axes[1,1].pcolormesh(μ_range, s_range, density)
axes[1,1].plot(true_params[1], true_params[0], color=red, marker='o',label="simulation parameter")
axes[1,1].plot(μ_est, s_est, color=green, marker='o', label="MAP estimate")
axes[1,1].legend(loc='lower left')

axes[2,0].plot(mut_samples, fitness_samples, ',k')
axes[2,0].plot(true_params[1], true_params[0], color=red, marker='o', label="simulation parameter")
axes[2,0].plot(μ_est, s_est, color=green, marker='o', label="MAP estimate")
axes[2,0].contour(μ_range, s_range, density, colors='k', linewidths=1)
axes[2,0].legend(loc='lower left')

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

f= open("est_real_params_delfi.csv","a+")
f.write(EvoModel+','+','.join(str(format(j)) for j in (true_params[0],s_est,true_params[1],μ_est,s_snv,m_snv)) + '\n')
f.close() 
