# -*- coding: utf-8 -*-
"""
Created on Feb 18 10:08:39 2021
cnv_sim_sbi

@author: grace
"""
import random
import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
import scipy.stats
import scipy.optimize
import scipy.interpolate
import seaborn as sns
import os
from PyPDF2 import PdfFileMerger

red, blue, green = sns.color_palette('Set1', 3)

from cnv_simulation import CNVsimulator_simpleWF, CNVsimulator_simpleChemo

red, blue, green = sns.color_palette('Set1', 3)

import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi
import torch

#### arguments ####
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-pd', "--presimulated_data")
parser.add_argument('-pt', "--presimulated_theta")
parser.add_argument('-obs', "--observations")
parser.add_argument('-o', "--out_file")
parser.add_argument('-s', "--seed")
parser.add_argument('-d', "--directory")
parser.add_argument('-g', "--generation_file")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
presim_data = str(args.presimulated_data)
presim_theta = str(args.presimulated_theta)
obs_name = str(args.observations)
out = str(args.out_file)
EvoModel = str(args.model)
path = str(args.directory)
g_file = str(args.generation_file)


#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3e8
s_snv=0.001 
m_snv=1e-5 
reps=1
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")
# chemostat parameters
A_inoculation = 1e5
S_init = .800
D=0.12
μA=0.45
k=.103
y=32445000*20
S0=.800
τ=1/10

#### prior ####
prior_min = np.log10(np.array([1e-4,1e-12]))
prior_max = np.log10(np.array([0.3,1e-3]))
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))


#### sbi simulator ####
def CNVsimulator(cnv_params):
    cnv_params = np.asarray(torch.squeeze(cnv_params,0))
    reps = 1
    if EvoModel == "WF":
        states = CNVsimulator_simpleWF(reps = reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, seed=None, parameters=cnv_params)
    if EvoModel == "Chemo":
        states = CNVsimulator_simpleChemo(reps = reps, A_inoculation=A_inoculation, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, S0=S0, y=y, τ=τ, seed=None, parameters=cnv_params)
        
    return states


#make sure simulator and prior adhere to sbi requirementst
simulator, prior = prepare_for_sbi(CNVsimulator, prior)

#### get presimulated data ####
theta_presimulated = torch.tensor(np.genfromtxt(path+'presimulated_data/'+presim_theta,delimiter=',')).type('torch.FloatTensor')
x_presimulated = torch.tensor(np.genfromtxt(path+'presimulated_data/'+presim_data,delimiter=',')).type('torch.FloatTensor')


#### run inference ####
inference = SNPE(prior, density_estimator='maf')
density_estimator = inference.append_simulations(theta_presimulated, x_presimulated).train()
posterior = inference.build_posterior(density_estimator)


#### functions for evaluating performance ####
# distance between a simulation and "observation" 
def param_distance(simulation, observation):
    # simulation and observation are both arrays of parameters of size 2
    simulation = np.power(10,simulation)
    observation = np.power(10, observation)
    d = ((simulation[0]-observation[0])**2 + (simulation[1]-observation[1])**2)**0.5
    return d

def distance(obs, exp):
    return np.sum((obs-exp)**2)

def format(value):
        return "%.12f" % value
    
def kernelminus(x):
    return -kernel(x)

def WAIC(logliks):
    S = logliks.size
    llpd = -np.log(S) + logsumexp(logliks)
    p1 = 2*(-np.log(S) + logsumexp(logliks) - logliks.mean())
    p2 = np.var(logliks, ddof=1)
    return -2*(llpd + -p1), -2*(llpd + -p2)

#### get all observed data ####
obs_file = np.genfromtxt(path + obs_name,delimiter=',')
# observed generations
gens=np.array([25,33,41,54,62,70,79,87,95,103,116,124,132,145,153,161,174,182,190,211,219,232,244,257,267])

#### for each observed data evaluate ####
for i in range(obs_file.shape[0]):
    observation = obs_file[i,0:25]
    true_params = torch.tensor(obs_file[i,25:27]).type('torch.FloatTensor')
    posterior_samples = posterior.sample((10000,), x=observation)
    log_probability = posterior.log_prob(posterior_samples, x=observation)
    
    fitness_samples = np.asarray(posterior_samples[:,0])
    mut_samples = np.asarray(posterior_samples[:,1])
    
    #calculations kde
    ymin = fitness_samples.min()
    ymax = fitness_samples.max()
    xmin = mut_samples.min()
    xmax = mut_samples.max()
    
    s_range, μ_range = np.mgrid[ymin:ymax:100j, xmin:xmax:100j]
    s_range = np.vstack((s_range, np.repeat(true_params[0], 100)))
    μ_range = np.vstack((μ_range, np.repeat(true_params[1], 100)))
    s_range = s_range[np.argsort(s_range[:, 0])]
    μ_range = μ_range[np.argsort(μ_range[:, 0])]
    positions = np.vstack([s_range.ravel(), μ_range.ravel()])
    values = np.vstack([fitness_samples, mut_samples])
    kernel = scipy.stats.gaussian_kde(values)
    density = np.reshape(kernel(positions).T, s_range.shape)
    
    #estimates for parameters from the posterior (MAP - highest probability in posterior)
    idx = np.argmax(density, axis=None)
    param_guess = np.array([positions[0,idx],positions[1,idx]])
    s_est, μ_est = scipy.optimize.minimize(kernelminus,param_guess, method ='Nelder-Mead', 
 options={'disp': True}).x
    map_dist = param_distance(np.array([s_est, μ_est]),true_params)
    map_log_prob = np.log(kernel([s_est, μ_est]))
    
    #draw from posterior
    indx=np.random.choice(posterior_samples.shape[0], 50, replace=False)
    params_post = posterior_samples[indx, :]
    # simulate based on drawn params, calculate distance
    reps=1
    dist_posterior = []
    evo_reps_posterior = []
    logprob_posterior = []
    for j in range(50):
        obs_post = CNVsimulator(params_post[j,:])
        evo_reps_posterior.append(obs_post)
        dist_posterior.append(distance(obs_post,observation))
        logprob_posterior.append(np.log(kernel(params_post[j,:])))
    evo_reps_posterior = np.vstack(evo_reps_posterior)
    dist_posterior = np.vstack(dist_posterior)
    logprob_posterior = np.vstack(logprob_posterior)
    
    # map posterior prediction
    post_prediction_map = CNVsimulator(torch.tensor([s_est, μ_est]))
    
    # aic and dic, waic
    aic = -2*map_log_prob + 2*2
    dic = 2*(np.log(kernel([fitness_samples.mean(), mut_samples.mean()])) - (1/50)*logprob_posterior.sum())
    waic1, waic2 = WAIC(np.log(kernel(posterior_samples.T)))
    #marginal 95% hdis
    fit_95hdi_low,fit_95hdi_high  = np.quantile(fitness_samples, q=[0.025,0.975])
    mut_95hdi_low,mut_95hdi_high  = np.quantile(mut_samples, q=[0.025,0.975])

    # save relevant values
    f= open(path + out + "_est_real_params.csv","a+")
    f.write(EvoModel+',SNPE,' + presim_data + ','.join(str(format(j)) 
                                      for j in
                                      (true_params[0],s_est,true_params[1],
                                       μ_est,s_snv,m_snv,map_dist,
                                       fit_95hdi_low,fit_95hdi_high,
                                       mut_95hdi_low,mut_95hdi_high,
                                       aic, dic, waic1, waic2)) + '\n')
    f.close() 
    
    # plots
    fig, axes = plt.subplots(3,2, figsize=(10, 10))
    
    
    # text description
    txt = 'SNPE\nModel: ' + EvoModel + '\nSimulation id: ' + str(i) + '\nlog10(CNV fitness effect): ' + str(str(true_params[0])) + '\nlog10(CNV mutation rate): ' + str(true_params[1]) + '\nSNV fitness:' + str(s_snv) + '\nSNV mutation rate:' + str(m_snv) + '\n' + presim_data
    
    axes[0,0].axis('off')
    axes[0,0].annotate(txt, (0.1, 0.5), xycoords='axes fraction', va='center')

    #observed
    axes[0,1].plot(gens,observation.reshape(-1),linewidth=4)
    axes[0,1].set(ylabel='Proportion of pop with CNV', xlabel='Generation')
    axes[0,1].set_title('Observed data')
    
    #posterior prediction
    for j in range(49) :
        axes[1,1].plot(gens, evo_reps_posterior[j,:], color='blue',alpha=0.1)
    axes[1,1].plot(gens, evo_reps_posterior[49,:], color='blue',alpha=0.1, 
                   label = 'posterior prediction')
    axes[1,1].plot(gens,observation.reshape(-1),linewidth=4, label = 'observation', color = 'black')
    axes[1,1].plot(gens,post_prediction_map.reshape(-1),linewidth=4, alpha=0.8, 
                   label = 'MAP posterior prediction', color='blue')
    axes[1,1].legend(loc='lower right')
    axes[1,1].set(xlabel='Generation',ylabel='Proportion of population with CNV')

    
    # marginal
    sns.distplot(fitness_samples, bins=50, kde=False, ax=axes[2,1])
    sns.distplot(mut_samples, bins=50, kde=False, ax=axes[1,0]) 
    #marginal 95% hdis
    axes[2,1].axvline(fit_95hdi_low, color='k', linestyle=':')
    axes[2,1].axvline(fit_95hdi_high, color='k', linestyle=':',label="95% HDI")
    axes[1,0].axvline(mut_95hdi_low, color='k', linestyle=':')
    axes[1,0].axvline(mut_95hdi_high, color='k', label="95% HDI", linestyle=':')
    # MAP and true params
    axes[2,1].axvline(true_params[0], color=red, label="simulation parameter")
    axes[2,1].axvline(s_est, color=green, label="MAP estimate")
    axes[1,0].axvline(true_params[1], color=red, label="simulation parameter")
    axes[1,0].axvline(μ_est, color=green, label="MAP estimate")
    axes[2,1].set(xlabel='log10(CNV fitness effect)')
    axes[1,0].set(xlabel='log10(CNV mutation rate)')
    axes[2,1].legend(loc='upper left')
    axes[1,0].legend(loc='upper left')
    
    # joint
    axes[2,0].set_xlim(μ_range.min(),μ_range.max())
    axes[2,0].set_ylim(s_range.min(),s_range.max())
    axes[2,0].pcolormesh(μ_range, s_range, density)
    CS = axes[2,0].contour(μ_range, s_range, density, 
                           levels = np.quantile(density, q=[0.5,0.95,0.99]),
                           colors=('w',),linestyles=('-',),linewidths=(2,))
    fmt = {}
    strs = [ '50%', '95%', '99%']
    for l,s in zip( CS.levels, strs ):
        fmt[l] = s
    axes[2,0].clabel(CS, CS.levels, fmt=fmt, inline=1, fontsize=15)
    axes[2,0].plot(true_params[1],true_params[0], color=red, marker='o', label="simulation parameter")
    axes[2,0].plot(μ_est, s_est, color="k", marker='o', label="MAP estimate")
    axes[2,0].legend(loc='lower left', prop={'size': 12})
    axes[2,0].set(xlabel='log10(CNV mutation rate)', ylabel='log10(CNV fitness effect)')
    
    fig.tight_layout()
    plt.title('')
    sns.despine()
    plt.savefig(path + str(i) + out + '.pdf', bbox_inches="tight")
    
x = [path + a for a in os.listdir(path) if a.endswith(out + ".pdf")]
merger = PdfFileMerger()
for pdf in x:
    merger.append(open(pdf, 'rb'))
with open(path + out + "_all.pdf", "wb") as fout:
    merger.write(fout)
    
