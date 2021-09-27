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
import pandas as pd

red, blue, green = sns.color_palette('Set1', 3)

from cnv_simulation import CNVsimulator_multiWF, CNVsimulator_multiChemo

import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi
import torch
import torch.nn as nn 
import torch.nn.functional as F 

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
generation=np.genfromtxt(g_file,delimiter=',', skip_header=1,dtype="int64")

#### prior ####
prior_min = np.array([0.5,np.log10(0.00001),np.log10(1e-12)])
prior_max = np.array([15,np.log10(0.8), np.log10(1e-3)])
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))


#### sbi simulator ####
reps = 8
def CNVsimulator(cnv_params):
    cnv_params = np.asarray(torch.squeeze(cnv_params,0))
    if model == "WF":
        states = CNVsimulator_multiWF(reps, N, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
    if model == "Chemo":
        states = CNVsimulator_multiChemo(reps, s_snv, m_snv, generation, parameters=cnv_params, seed=None)
        
    return states

#embedding net
embedding_net = nn.Sequential(
    nn.Linear(200, 25), 
    nn.ReLU(),
    nn.Linear(25, 20), 
    nn.ReLU(),
    nn.Linear(20, 5)) 

#make sure simulator and prior adhere to sbi requirementst
simulator, prior = prepare_for_sbi(CNVsimulator, prior)

#### get presimulated data ####
theta_presimulated = torch.tensor(np.genfromtxt(path+'presimulated_data_multi/'+presim_theta,delimiter=',')).type('torch.FloatTensor')
x_presimulated = torch.tensor(np.genfromtxt(path+'presimulated_data_multi/'+presim_data,delimiter=',')).type('torch.FloatTensor')


# neural_posterior = utils.posterior_nn(model='maf', 
neural_posterior = utils.posterior_nn(model='nsf', 
                                      embedding_net=embedding_net)

# setup the inference procedure with the SNPE
inference = SNPE(prior=prior, density_estimator=neural_posterior)

density_estimator = inference.append_simulations(theta_presimulated, x_presimulated).train()
posterior = inference.build_posterior(density_estimator) 

#### functions for evaluating performance ####
def rmse(a, b):
    return ((a-b)**2).mean()**0.5

def statistics_rmse(a, b):
    x=[]
    for i in range(a.shape[0]):
#         for j in range(b.shape[0]):
#             x.append(rmse([a[i,:], b[j,:]]))
        x.append(rmse(a[i,:],b))
    x = np.array(x)
    x = x[~np.isnan(x)]
    return np.mean(x),scipy.stats.norm.interval(alpha=0.95, loc=np.mean(x), scale=scipy.stats.sem(x))

def statistics_multi_rmse(a, b):
    x=[]
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            x.append(rmse(a[i,:], b[j,:]))
    x = np.array(x)
    x = x[~np.isnan(x)]
    return np.mean(x),scipy.stats.norm.interval(alpha=0.95, loc=np.mean(x), scale=scipy.stats.sem(x))

def statistics_corrcoef(ppc,observation):
    x=np.append(observation.reshape(1,observation.shape[0]), ppc, 0)
    corrs = np.corrcoef(x)[0,1:]
    corrs = corrs[~np.isnan(corrs)]
    return np.nanmean(corrs),scipy.stats.norm.interval(alpha=0.95, loc=np.mean(corrs), scale=scipy.stats.sem(corrs))


def statistics_multi_corrcoef(ppc,observation):
    corrs = np.array([])
    for i in range(observation.shape[0]):
        x=np.append(observation[i,:].reshape(1,observation[i,:].shape[0]), ppc, 0)
        corrs = np.append(corrs,np.corrcoef(x)[0,1:])
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

def var_set(x):
    x = x.T
    dx = x[1:] - x[:-1]
    return dx.var()

def kde_2d(x,y):
    X,Y= np.mgrid[x.min():x.max():50j, y.min():y.max():50j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = scipy.stats.gaussian_kde(values)
    density = np.reshape(kernel(positions).T, X.shape)
    return X,Y,density    
    
#####
def evaluate_post(obs_name, observation, left_out, lo_ids, true_params):
    posterior_samples = posterior.sample((5000,), x=observation.reshape(-1))
    log_probability = posterior.log_prob(posterior_samples, x=observation.reshape(-1))
    
#     posterior_samples = posterior.sample((5000,), x=np.apply_along_axis(lambda x: np.histogram(x, bins=1000, range=(0,1))[0]/x.size, 1, observation).reshape(1,8000))
#     log_probability = posterior.log_prob(posterior_samples, x=np.apply_along_axis(lambda x: np.histogram(x, bins=1000, range=(0,1))[0]/x.size, 1, observation).reshape(1,8000))
    
    shape_samples = np.asarray(posterior_samples[:,0])
    scale_samples = np.asarray(posterior_samples[:,1])
    mut_samples = np.asarray(posterior_samples[:,2])
    
    #calculations kde
    xmin = shape_samples.min()
    xmax = shape_samples.max()
    ymin = scale_samples.min()
    ymax = scale_samples.max()
    zmin = mut_samples.min()
    zmax = mut_samples.max()

    shape_range, scale_range, μ_range = np.mgrid[xmin:xmax:50j, ymin:ymax:50j,zmin:zmax:50j]
    shape_range = shape_range[np.argsort(shape_range[:, 0])]
    scale_range = scale_range[np.argsort(scale_range[:, 0])]
    μ_range = μ_range[np.argsort(μ_range[:, 0])]
    positions = np.vstack([shape_range.ravel(), scale_range.ravel(), μ_range.ravel()])
    values = np.vstack([shape_samples, scale_samples, mut_samples])
    kernel = scipy.stats.gaussian_kde(values)
    density = np.reshape(kernel(positions).T, shape_range.shape)

    #estimates for parameters from the posterior (MAP - highest probability in posterior)
    idx = np.argmax(density, axis=None)
    param_guess = np.array([positions[0,idx],positions[1,idx],positions[2,idx]])
    def kernelminus(x):
        return -kernel(x)
    shape_est, scale_est, μ_est = scipy.optimize.minimize(kernelminus,param_guess, method ='Nelder-Mead',
                                                          options={'disp': True}).x
    map_log_prob = np.log(kernel([shape_est, scale_est, μ_est]))

    shape_95hdi_low,shape_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=shape_samples, weights=None)
    scale_95hdi_low,scale_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=scale_samples, weights=None)
    mut_95hdi_low,mut_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=mut_samples, weights=None)
    
    #draw from posterior
    indx=np.random.choice(posterior_samples.shape[0], 50, replace=False)
    params_post = posterior_samples[indx, :]
    # simulate based on drawn params, calculate distance
    evo_reps_posterior = []
    logprob_posterior = []
    for j in range(50):
        obs_post = CNVsimulator(params_post[j,:])
        evo_reps_posterior.append(obs_post)
        logprob_posterior.append(np.log(kernel(params_post[j,:])))
    evo_reps_posterior = np.vstack(evo_reps_posterior)
    logprob_posterior = np.vstack(logprob_posterior)

    # rmse and correlation
    mean_rmse_ppc, rmse95_ci = statistics_multi_rmse(evo_reps_posterior,observation)
    mean_corr_ppc, corr95_ci = statistics_multi_corrcoef(evo_reps_posterior,observation)
    rmse95_low_ppc,rmse95_hi_ppc = rmse95_ci
    corr95_low_ppc, corr95_hi_ppc = corr95_ci

    # map posterior prediction
    post_prediction_map = CNVsimulator(torch.tensor([shape_est, scale_est, μ_est]))
    aic = -2*map_log_prob + 2*3
    dic = 2*(np.log(kernel([shape_samples.mean(), scale_samples.mean(), mut_samples.mean()])) - (1/50)*logprob_posterior.sum())
    waic1, waic2 = WAIC(np.log(kernel(posterior_samples.T)))
    var_observation = var_set(np.append(observation,left_out,0))
    var_ppc = var_set(evo_reps_posterior)
    
    # save relevant values
    f= open(path + out + "_est_real_params.csv","a+")
    f.write(obs_name + ',' + lo_ids + ',' + model+',NPE,' + presim_data + ','.join(str(format(j)) 
                                      for j in
                                      (true_params[0],shape_est,true_params[1],scale_est,true_params[2],
                                       μ_est,s_snv,m_snv,mean_rmse_ppc,rmse95_low_ppc,rmse95_hi_ppc,
                                       mean_corr_ppc,corr95_low_ppc, corr95_hi_ppc,
                                       shape_95hdi_low,shape_95hdi_high,scale_95hdi_low,scale_95hdi_high,
                                       mut_95hdi_low,mut_95hdi_high,
                                       aic, dic, waic1, waic2,var_observation, var_ppc)) + '\n')
    f.close() 
    
    # plots
    fig, axes = plt.subplots(3,3, figsize=(14, 14))
    
    # text description
    # somewhere 
    txt = 'NPE\nModel: ' + model + '\nSet ID: \n' + str(obs_name) + '\nleft out IDs: ' + lo_ids + '\ngamma shape: ' + str(str(true_params[0])) + '\nlog10(gamma scale): ' + str(true_params[1]) +'\nlog10(CNV mutation rate): ' + str(true_params[2]) +'\nSNV fitness:' + str(s_snv) + '\nSNV mutation rate:' + str(m_snv) + '\n' + presim_data
    axes[0,1].axis('off')
    axes[0,1].annotate(txt, (-0.1, 0.5), xycoords='axes fraction', va='center')
    
    #observed
    axes[0,2].plot(generation,observation.reshape(8,25)[1:,:].T,linewidth=1, color = 'black', alpha=0.6)
    axes[0,2].plot(generation,observation.reshape(8,25)[0,:],linewidth=1, color = 'black', alpha=0.6, label = 'used observations')
    axes[0,2].plot(generation,left_out[1:,:].T,linewidth=1, color = 'purple', alpha=0.6)
    axes[0,2].plot(generation,left_out[0,:],linewidth=1, color = 'purple', alpha=0.6, label = 'held out observations')
    axes[0,2].legend(loc='lower right')
    axes[0,2].set(ylabel='Proportion of pop with CNV', xlabel='Generation')
    axes[0,2].set_title('Observed data')

    #posterior prediction
    axes[1,2].plot(generation, evo_reps_posterior.T, color='blue',alpha=0.1)
    axes[1,2].plot(generation, evo_reps_posterior[49,:], color='blue',alpha=0.1,
               label = 'posterior prediction')
    axes[1,2].plot(generation,observation.reshape(8,25)[1:,:].T,linewidth=1, color = 'black', alpha=0.6)
    axes[1,2].plot(generation,observation.reshape(8,25)[0,:],linewidth=1, color = 'black', alpha=0.6, label = 'observation')
    axes[1,2].plot(generation,left_out[1:,:].T,linewidth=1, color = 'purple', alpha=0.6)
    axes[1,2].plot(generation,left_out[0,:],linewidth=1, color = 'purple', alpha=0.6, label = 'held out observations')
    axes[1,2].plot(generation,post_prediction_map.T,linewidth=4, alpha=0.8, 
               label = 'MAP posterior prediction', color='blue')
    axes[1,2].legend(loc='lower right')
    axes[1,2].set(xlabel='Generation',ylabel='Proportion of population with CNV')
    
    # marginal
    sns.distplot(shape_samples, bins=50, kde=False, ax=axes[0,0])
    sns.distplot(scale_samples, bins=50, kde=False, ax=axes[1,1])
    sns.distplot(mut_samples, bins=50, kde=False, ax=axes[2,2]) 
    #marginal 95% hdis
    axes[0,0].axvline(shape_95hdi_low, color='k', linestyle=':')
    axes[0,0].axvline(shape_95hdi_high, color='k', linestyle=':',label="95% HDI")
    axes[1,1].axvline(scale_95hdi_low, color='k', linestyle=':')
    axes[1,1].axvline(scale_95hdi_high, color='k', linestyle=':',label="95% HDI")
    axes[2,2].axvline(mut_95hdi_low, color='k', linestyle=':')
    axes[2,2].axvline(mut_95hdi_high, color='k', label="95% HDI", linestyle=':')

    # MAP and true params
    axes[0,0].axvline(true_params[0], color=red, label="simulation parameter")
    axes[0,0].axvline(shape_est, color=green, label="MAP estimate")
    axes[1,1].axvline(true_params[1], color=red, label="simulation parameter")
    axes[1,1].axvline(scale_est, color=green, label="MAP estimate")
    axes[2,2].axvline(true_params[2], color=red, label="simulation parameter")
    axes[2,2].axvline(μ_est, color=green, label="MAP estimate")
    axes[0,0].set(xlabel='gamma shape')
    axes[1,1].set(xlabel='log10(gamma scale)')
    axes[2,2].set(xlabel='log10(CNV mutation rate)')
    axes[0,0].legend(loc='upper left')
    axes[1,1].legend(loc='upper left')
    axes[2,2].legend(loc='upper left')
    
    # pairwise marginals
    X,Y,Z = kde_2d(shape_samples, scale_samples)
    axes[1,0].pcolormesh(X,Y,Z)
    X,Y,Z = kde_2d(scale_samples, mut_samples)
    axes[2,1].pcolormesh(X,Y,Z)
    X,Y,Z = kde_2d(shape_samples, mut_samples)
    axes[2,0].pcolormesh(X,Y,Z)

    axes[1,0].plot(true_params[0],true_params[1], color=red, marker='o', label="simulation parameter")
    axes[1,0].plot(shape_est, scale_est, color="k", marker='o', label="MAP estimate")

    axes[2,1].plot(true_params[1],true_params[2], color=red, marker='o', label="simulation parameter")
    axes[2,1].plot(scale_est, μ_est, color="k", marker='o', label="MAP estimate")

    axes[2,0].plot(true_params[0],true_params[2], color=red, marker='o', label="simulation parameter")
    axes[2,0].plot(shape_est, μ_est, color="k", marker='o', label="MAP estimate")

    axes[2,0].legend(loc='lower left', prop={'size': 12})
    axes[1,0].set(xlabel='gamma shape', ylabel='log10(gamma scale)')
    axes[2,1].set(xlabel='log10(gamma scale)', ylabel='log10(mutation rate)')
    axes[2,0].set(xlabel='gamma shape', ylabel='log10(mutation rate)')
    plt.title('')
    sns.despine()

    return fig

#### get all observed data ####
obs_files = pd.read_csv(path + obs_name, header=None)
leave_outs = np.array([[0,3,6],[1,4,7],[2,5,8]])
#### for each observed data evaluate ####
reps=1 # all post pred will only have one rep
for index, row in obs_files.iterrows():
    all_obs = np.genfromtxt(path+row[0], delimiter=',')[:,0:25]
    true_params = np.array([row[0].split("_")[1].replace("shape",""),
                        row[0].split("_")[2].replace("scale",""),
                        "-" + row[0].split("_")[3].replace("mut","")], dtype=float)
    true_params[1] = np.log10(true_params[1])
    for j in leave_outs:
        observation = np.delete(all_obs, j, 0)
        left_out = all_obs[j,:]
        fig = evaluate_post(obs_name=row[0], observation=observation, left_out=left_out, lo_ids=str(j), true_params=true_params)
        plt.gcf().tight_layout()
        plt.savefig(path + row[0] + str(j[0]) + out + '.pdf', bbox_inches="tight")


x = [path + a for a in os.listdir(path) if a.endswith(out + ".pdf")]
merger = PdfFileMerger()
for pdf in x:
    merger.append(open(pdf, 'rb'))
with open(path + out + "_all.pdf", "wb") as fout:
    merger.write(fout)
    
