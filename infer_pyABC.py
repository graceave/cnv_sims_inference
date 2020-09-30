# -*- coding: utf-8 -*-
"""
Created on August 2020
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
import os
import tempfile
import logging

from pyabc import ABCSMC, RV, Distribution
from pyabc.distance import AdaptivePNormDistance, PercentileDistance

#from pyabc.sampler import SingleCoreSampler
from pyabc.visualization import plot_kde_1d, plot_kde_2d

from cnv_simulation import CNVsimulator_simpleWF, CNVsimulator_simpleChemo
red, blue, green = sns.color_palette('Set1', 3)

#### arguments ####
# infer_pyABC.py -m "WF" -cs 0.001 -cu 1e-7 -ss 0.001 -su 1e-5 -s $SLURM_ARRAY_TASK_ID -o "snvWSSM_cnvWSWM$SLURM_ARRAY_TASK_ID"
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-cs', "--cnv_fitness_effect")
parser.add_argument('-cu', "--cnv_mutation_rate")
parser.add_argument('-ss', "--snv_fitness_effect")
parser.add_argument('-su', "--snv_mutation_rate")
parser.add_argument('-s', "--seed")
parser.add_argument('-o', "--outplot")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
outfile = str(args.outplot)
model = str(args.model)
true_params = np.log10(np.array([float(args.cnv_fitness_effect), float(args.cnv_mutation_rate)]))  
s_snv = float(args.snv_fitness_effect)
m_snv = float(args.snv_mutation_rate)

# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3e8
generation = np.array(range(0,268))
seed_true = 1
# chemostat parameters
A_inoc = 1e5
S_init = .800
D=0.12
μA=0.45
k=.103
y=3244500
I=.800
τ=1/10

# true parameters from arguments
labels_params = ['CNV fitness effect', 'CNV mutation rate']

# observed data
if model == "WF":
    data_observed = CNVsimulator_simpleWF(N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, seed=argseed, cnv_params=true_params)
if model == "Chemo":
    data_observed = CNVsimulator_simpleChemo(A_inoc=A_inoc, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, I=I, y=y, τ=τ, cnv_params=true_params)
exp_gen = ['25', '33', '41', '54', '62', '70', '79', '87', '95', '103', '116',
       '124', '132', '145', '153', '161', '174', '182', '190', '211',
       '219', '232', '244', '257', '267']
dict_observed = {}
i = 0
for keys in exp_gen: 
    dict_observed[keys] = data_observed[i]
    i+=1
    
# simulation wrapper for pyABC
def simulate_pyabc(parameters, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, model=model):
    if model == "WF":
        res = CNVsimulator_simpleWF(N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, parameters=parameters)
    if model == "Chemo":
        res = CNVsimulator_simpleChemo(A_inoc=A_inoc, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, I=I, y=y, τ=τ, parameters=parameters)
    exp_gen = ['25', '33', '41', '54', '62', '70', '79', '87', '95', '103', '116',
       '124', '132', '145', '153', '161', '174', '182', '190', '211',
       '219', '232', '244', '257', '267']
    pyabc_dict = {}
    i = 0
    for keys in exp_gen: 
        pyabc_dict[keys] = res[i]
        i+=1
    return pyabc_dict

# prior
prior = Distribution(
    s=RV("uniform", np.log10(1e-4), np.log10(0.3)-np.log10(1e-4)),
    m=RV("uniform", np.log10(1e-12), np.log10(1e-3)-np.log10(1e-12))
)

# abc smc object and run
abc = ABCSMC(models=simulate_pyabc,
             parameter_priors=prior,
             distance_function=AdaptivePNormDistance(p=2),
            # sampler=SingleCoreSampler(),
             population_size=10000)
db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "WF.db"))
history = abc.new(db_path, dict_observed)
history = abc.run(minimum_epsilon=0.2, max_nr_populations=10)

# get posterior
params, weights = history.get_distribution(0)

#### plots to output to pdf ####
# for plotting
gens = np.array([25,33,41,54,62,70,79,87,95,103,116,124,132,145,153,161,174,182,190,211,219,232,244,257,267])

prior_min = np.log10(np.array([1e-4,1e-12]))
prior_max = np.log10(np.array([0.3,1e-3]))

s_range, μ_range = np.mgrid[np.log10(1e-4):np.log10(0.3):100j, np.log10(1e-12):np.log10(1e-3):100j]
positions = np.vstack([s_range.ravel(), μ_range.ravel()])
values = np.vstack([params['s'], params['m']])
kernel = scipy.stats.gaussian_kde(values)
density = np.reshape(kernel(positions).T, s_range.shape)


#estimates for parameters from the posterior (MAP - highest probability in posterior)
idx = np.argmax(density, axis=None)
param_guess = np.array([positions[0,idx],positions[1,idx]])
def kernelminus(x):
    return -kernel(x)
s_est, μ_est = scipy.optimize.minimize(kernelminus,param_guess, method ='Nelder-Mead', 
 options={'disp': True}).x


fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# observed
axes[0,0].plot(gens,data_observed)
axes[0,0].set(ylabel='CNV frequency', xlabel='Generation')
axes[0,0].set_title('observed data')


# marginal
sns.distplot(params['s'], bins=50, kde=False, ax=axes[1,0]) 
sns.distplot(params['m'], bins=50, kde=False, ax=axes[0,1]) 

axes[1,0].axvline(true_params[0], color=red, label="simulation parameter")
axes[1,0].axvline(s_est, color=green,label="MAP estimate")
axes[1,0].set(xlabel='log10(CNV fitness effect)')
axes[1,0].legend()
axes[0,1].axvline(true_params[1], color=red,label="simulation parameter")
axes[0,1].axvline(μ_est, color=green,label="MAP estimate")
axes[0,1].set(xlabel='log10(CNV mutation rate)')
axes[0,1].legend()

# joint
axes[1,1].pcolormesh(μ_range, s_range, density)
axes[1,1].plot(true_params[1], true_params[0], color=red, marker='o',label="simulation parameter")
axes[1,1].plot(μ_est, s_est, color=green, marker='o', label="MAP estimate")
axes[1,1].legend(loc='lower left')
axes[1,1].set(xlabel='log10(CNV mutation rate)', ylabel='log10(CNV fitness effect)')

fig.tight_layout()
plt.title('model:'+ model +'\nlog10(CNV fitness): ' + str(true_params[0]) + '\nlog10(CNV mutation rate): ' + 
                    str(true_params[1]) + 'SNV fitness:' + str(s_snv) + 'SNV mutation rate:' + str(m_snv))
sns.despine()
plt.savefig(outfile + '.pdf')  

#### Write out the estimated parameters and "true" parameters ####
# single file for all combos
def format(value):
    return "%.12f" % value

f= open("est_real_params_pyABC.csv","a+")
f.write(model+','+','.join(str(format(j)) for j in (true_params[0],s_est,true_params[1],μ_est,s_snv,m_snv)) + '\n')
f.close() 
