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
import seaborn as sns
import os
from PyPDF2 import PdfFileMerger
from scipy.special import logsumexp
import pyabc.visualization

from cnv_simulation import CNVsimulator_simpleWF, CNVsimulator_simpleChemo

red, blue, green = sns.color_palette('Set1', 3)

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
model = str(args.model)
path = str(args.directory)
g_file = str(args.generation_file)


#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3.3e8
s_snv=0.001 
m_snv=1e-5 
reps=1
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")


#### get presimulated data ####
theta_presimulated = np.genfromtxt(path+'presimulated_data/'+presim_theta,delimiter=',')
x_presimulated = np.genfromtxt(path+'presimulated_data/'+presim_data,delimiter=',')

#### functions for evaluating performance ####
# root mean square error between simulated trajectory and "true" trajectory
def rmse(a, b):
    return ((a-b)**2).mean()**0.5

def statistics_rmse(a, b):
    x=[]
    for i in range(a.shape[0]):
        x.append(rmse(a[i,:],b))
    return np.mean(x),scipy.stats.norm.interval(alpha=0.95, loc=np.mean(x), scale=scipy.stats.sem(x))

def statistics_corrcoef(ppc,observation):
    x=np.append(observation.reshape(1,observation.shape[0]), ppc, 0)
    corrs = np.corrcoef(x)[0,1:]
    corrs = corrs[~np.isnan(corrs)]
    return np.nanmean(corrs),scipy.stats.norm.interval(alpha=0.95, loc=np.mean(corrs), scale=scipy.stats.sem(corrs))

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

def get_hdr(density, mylevel): #this function is thanks to the lovely folks who wrote sbi
    shape = density.shape
    probs = density.flatten()
    levels = np.asarray([mylevel])#,0.05, 0.5])
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)
    
    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]
    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]
    # create contours at level
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)
    return contours

# distance for rejection ABC
def distance_euc(sim, obs):
    d=((sim-obs)**2).sum(axis=1)
    return d**0.5

#### get all observed data ####
obs_file = np.genfromtxt(path + obs_name,delimiter=',')

#### for each observed data evaluate ####
for i in range(obs_file.shape[0]):
    observation = obs_file[i,0:25]
    true_params = obs_file[i,25:27]
    
    # get distances
    distances = distance_euc(x_presimulated, observation)
    # threshold that accepts 5% of the samples
    quantile = 0.05
    ϵ_e = np.quantile(distances, quantile)
    # get accepted
    idx_accepted = distances < ϵ_e
    
    posterior_samples = theta_presimulated[idx_accepted, ]
    
    fitness_samples = np.asarray(posterior_samples[:,0])
    mut_samples = np.asarray(posterior_samples[:,1])
    
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
    s_est, μ_est = scipy.optimize.minimize(kernelminus,param_guess, method ='Nelder-Mead', 
 options={'disp': True}).x
    map_log_prob = np.log(kernel([s_est, μ_est]))
    
    #draw from posterior
    indx=np.random.choice(posterior_samples.shape[0], 50, replace=False)
    params_post = posterior_samples[indx, :]
    # simulate based on drawn params, calculate distance
    reps=1
    evo_reps_posterior = []
    logprob_posterior = []
    for j in range(50):
        if model == "WF":
            obs_post = CNVsimulator_simpleWF(reps = reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, seed=None, parameters=params_post[j,:])
        if model == "Chemo":
            obs_post = CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters=params_post[j,:], seed=None)
        evo_reps_posterior.append(obs_post)
        logprob_posterior.append(np.log(kernel(params_post[j,:])))
    evo_reps_posterior = np.vstack(evo_reps_posterior)
    logprob_posterior = np.vstack(logprob_posterior)
    
    mean_rmse_ppc, rmse95_ci = statistics_rmse(evo_reps_posterior,observation)
    mean_corr_ppc, corr95_ci = statistics_corrcoef(evo_reps_posterior,observation)
    rmse95_low_ppc,rmse95_hi_ppc = rmse95_ci
    corr95_low_ppc, corr95_hi_ppc = corr95_ci
    
    # map posterior prediction
    if model == "WF":
        post_prediction_map = CNVsimulator_simpleWF(reps = reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, seed=None, parameters=np.array([s_est, μ_est]))
    if model == "Chemo":
        post_prediction_map = CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters=np.array([s_est, μ_est]))
 
    rmse_map = rmse(post_prediction_map, observation)
    corr_map = np.corrcoef(np.append(observation.reshape(1,observation.shape[0]), post_prediction_map, 0))[0,1]
    
    # aic and dic, waic
    aic = -2*map_log_prob + 2*2
    dic = 2*(np.log(kernel([fitness_samples.mean(), mut_samples.mean()])) - (1/50)*logprob_posterior.sum())
    waic1, waic2 = WAIC(np.log(kernel(posterior_samples.T)))
    #marginal 95% hdis
    fit_95hdi_low,fit_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=fitness_samples, weights=None)
    mut_95hdi_low,mut_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=mut_samples, weights=None)
    # save relevant values
    f= open(path + out + "_est_real_params.csv","a+")
    f.write(model+',rejectionABC,' + presim_data + ','.join(str(format(j)) 
                                      for j in
                                      (true_params[0],s_est,true_params[1],
                                       μ_est,s_snv,m_snv,rmse_map,
                                       mean_rmse_ppc,rmse95_low_ppc,rmse95_hi_ppc,
                                       corr_map,
                                       mean_corr_ppc,corr95_low_ppc, corr95_hi_ppc,
                                       fit_95hdi_low,fit_95hdi_high,
                                       mut_95hdi_low,mut_95hdi_high,
                                       aic, dic, waic1, waic2)) + '\n')
    f.close() 
    
    # plots
    fig, axes = plt.subplots(3,2, figsize=(10, 10))
    
    
    # text description
    txt = 'Rejection ABC\nModel: ' + model + '\nSimulation id: ' + str(i) + '\nlog10(CNV fitness effect): ' + str(true_params[0]) + '\nlog10(CNV mutation rate): ' + str(true_params[1]) + '\nSNV fitness:' + str(s_snv) + '\nSNV mutation rate:' + str(m_snv) + '\n' + presim_data
    
    axes[0,0].axis('off')
    axes[0,0].annotate(txt, (0.1, 0.5), xycoords='axes fraction', va='center')

    #observed
    axes[0,1].plot(generation, observation.reshape(-1),linewidth=4)
    axes[0,1].set(ylabel='Proportion of pop with CNV', xlabel='Generation')
    axes[0,1].set_title('Observed data')
    
    #posterior prediction
#     for j in range(49) :
#         axes[1,1].plot(generation, evo_reps_posterior[j,:], color='blue',alpha=0.1)
#     axes[1,1].plot(generation, evo_reps_posterior[49,:], color='blue',alpha=0.1,
    axes[1,1].plot(generation, evo_reps_posterior.T, color='blue',alpha=0.1)
    axes[1,1].plot(generation, evo_reps_posterior[49,:], color='blue',alpha=0.1, 
                   label = 'posterior prediction')
    axes[1,1].plot(generation,observation.reshape(-1),linewidth=4, label = 'observation', color = 'black')
    axes[1,1].plot(generation,post_prediction_map.reshape(-1),linewidth=4, alpha=0.8, 
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
#     axes[2,0].set_xlim(μ_range.min(),μ_range.max())
#     axes[2,0].set_ylim(s_range.min(),s_range.max())
    # #get values from contour
    axes[2,0].pcolormesh(μ_range, s_range, density)
    CS2=axes[2,0].contour(μ_range, s_range,get_hdr(density, mylevel=[0.95]), colors='white')
    CS3=axes[2,0].contour(μ_range, s_range,get_hdr(density, mylevel=[0.5]), colors='lightgrey')
    axes[2,0].clabel(CS2, fmt='0.95')
    axes[2,0].clabel(CS3, fmt='0.5')
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
    
