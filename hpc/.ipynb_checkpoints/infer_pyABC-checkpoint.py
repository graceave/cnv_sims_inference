# -*- coding: utf-8 -*-
"""
Created on August 2020
cnv_sim_delfi

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
import tempfile
from PyPDF2 import PdfFileMerger
from pyabc.sampler import ConcurrentFutureSampler
from concurrent.futures import ProcessPoolExecutor#ThreadPoolExecutor

red, blue, green = sns.color_palette('Set1', 3)


from pyabc import ABCSMC, RV, Distribution
from pyabc.distance import AdaptivePNormDistance
import pyabc.distance
from pyabc.populationstrategy import AdaptivePopulationSize
import pyabc.visualization


#from pyabc.sampler import MulticoreParticleParallelSampler, MulticoreEvalParallelSampler
from pyabc.sge import nr_cores_available

from cnv_simulation import CNVsimulator_simpleWF, CNVsimulator_simpleChemo

#### arguments ####

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model")
parser.add_argument('-obs', "--observations")
parser.add_argument('-p', "--starting_particle_size")
parser.add_argument('-s', "--seed")
parser.add_argument('-o', "--out_file")
parser.add_argument('-d', "--directory")
parser.add_argument('-c', "--cores")
parser.add_argument('-g', "--generation_file")
args = parser.parse_args()

argseed = int(args.seed)
random.seed(int(argseed))
path = str(args.directory)
out = str(args.out_file)
obs_name = str(args.observations)
particle_size = int(args.starting_particle_size)
model = str(args.model)
n_cores = int(args.cores)
g_file = str(args.generation_file)

# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3e8
s_snv=0.001
m_snv=1e-5
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
    
# simulation wrapper for pyABC
reps=1
def simulate_pyabc(parameters):
    if model == "WF":
        res = CNVsimulator_simpleWF(reps=reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, parameters=parameters)
    if model == "Chemo":
        res = CNVsimulator_simpleChemo(reps=reps, A_inoculation=A_inoculation, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, S0=S0, y=y, τ=τ, parameters=parameters)
    exp_gen = ['25', '33', '41', '54', '62', '70', '79', '87', '95', '103', '116',
       '124', '132', '145', '153', '161', '174', '182', '190', '211',
       '219', '232', '244', '257', '267']
    pyabc_dict = {}
    i = 0
    for keys in exp_gen: 
        pyabc_dict[keys] = np.array(res[:,i].item())
        i+=1
    return pyabc_dict

# prior
prior = Distribution(
    s=RV("uniform", np.log10(1e-4), np.log10(0.3)-np.log10(1e-4)),
    m=RV("uniform", np.log10(1e-12), np.log10(1e-3)-np.log10(1e-12))
)

# abc smc object
pool = ProcessPoolExecutor(max_workers=n_cores)
sampler = ConcurrentFutureSampler(pool)
abc = ABCSMC(models=simulate_pyabc,
             parameter_priors=prior,
             distance_function=AdaptivePNormDistance(p=2, scale_function=pyabc.distance.root_mean_square_deviation),
             sampler=sampler,
             population_size=AdaptivePopulationSize(start_nr_particles=particle_size))


#functions
# distance between a simulation and "observation" 
def param_distance(simulation, observation):
    # simulation and observation are both arrays of parameters of size 2
    simulation = np.power(10,simulation)
    observation = np.power(10, observation)
    d = ((simulation[0]-observation[0])**2 + (simulation[1]-observation[1])**2)**0.5
    return d

def kernelminus(x):
    return -kernel(x)

def distance(obs, exp):
    return np.sum((obs-exp)**2)

def format(value):
        return "%.12f" % value

def WAIC(logliks):
    S = logliks.size
    llpd = -np.log(S) + logsumexp(logliks)
    p1 = 2*(-np.log(S) + logsumexp(logliks) - logliks.mean())
    p2 = np.var(logliks, ddof=1)
    return -2*(llpd + -p1), -2*(llpd + -p2)
    
#### get all observed data ####
obs_file = np.genfromtxt(path + obs_name,delimiter=',')

exp_gen = ['25', '33', '41', '54', '62', '70', '79', '87', '95', '103', '116', '124', '132', '145', '153', '161', '174', '182', '190', '211','219', '232', '244', '257', '267']
gens=generation

for i in range(obs_file.shape[0]):
    observation = obs_file[i,0:25]
    true_params = obs_file[i,25:27]
    labels_params = ['CNV fitness effect', 'CNV mutation rate']
    dict_observed = {}
    j = 0
    for keys in exp_gen:
        dict_observed[keys] = np.array(observation[j])
        j+=1
    # and we define where to store the results
    db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), model + str(particle_size) + str(i) + "_pyABC_run.db"))
    history = abc.new(db_path, dict_observed)
    print("ABC-SMC run ID:", history.id)
    # We run the ABC until either criterion is met
    history = abc.run(minimum_epsilon=0.02, max_nr_populations=10) 
    
    # get posterior
    # to numpy array
    params, weights = history.get_distribution()
    fitness_samples = np.asarray(params['s'])
    mut_samples = np.asarray(params['m'])
    #calculations kde
    ymin = params['s'].min()
    ymax = params['s'].max()
    xmin = params['m'].min()
    xmax = params['m'].max()
    
    s_range, μ_range = np.mgrid[ymin:ymax:100j, xmin:xmax:100j]
    positions = np.vstack([s_range.ravel(), μ_range.ravel()])
    values = np.vstack([params['s'], params['m']])
    kernel = scipy.stats.gaussian_kde(values, weights=weights)
    density = np.reshape(kernel(positions).T, s_range.shape)
    
    #estimates for parameters from the posterior (MAP - highest probability in posterior)
    idx = np.argmax(density, axis=None)
    param_guess = np.array([positions[0,idx],positions[1,idx]])
    s_est, μ_est = scipy.optimize.minimize(kernelminus,param_guess, method ='Nelder-Mead', 
 options={'disp': True}).x
    map_dist = param_distance(np.array([s_est, μ_est]),true_params)
    map_log_prob = np.log(kernel([s_est, μ_est]))
    
    #draw from posterior
    idx = np.random.choice(params.shape[0], 50, replace=False)
    params_post = np.array([np.asarray(fitness_samples[idx]), np.asarray(mut_samples[idx])]).T
    
    #simulate based on drawn params, calculate distance, calculate probability
    dist_posterior = []
    evo_reps_posterior = []
    logprob_posterior = []
    for j in range(50):
        if model == "WF":
            obs_post = CNVsimulator_simpleWF(reps=reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, parameters=params_post[j,:])
        if model == "Chemo":
            obs_post = CNVsimulator_simpleChemo(reps=reps, A_inoculation=A_inoculation, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, S0=S0, y=y, τ=τ, parameters=params_post[j,:])
        evo_reps_posterior.append(obs_post)
        dist_posterior.append(distance(obs_post,observation))
        logprob_posterior.append(np.log(kernel(params_post[j,:])))
    evo_reps_posterior = np.vstack(evo_reps_posterior)
    dist_posterior = np.vstack(dist_posterior)
    logprob_posterior = np.vstack(logprob_posterior)
    
    if model == "WF":
        post_prediction_map = CNVsimulator_simpleWF(reps=reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, parameters=np.array([s_est, μ_est]))
    if model == "Chemo":
        post_prediction_map = CNVsimulator_simpleChemo(reps=reps, A_inoculation=A_inoculation, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, S0=S0, y=y, τ=τ, parameters=np.array([s_est, μ_est]))
    
    #aic and dic
    aic = -2*map_log_prob + 2*2
    dic = 2*(np.log(kernel([fitness_samples.mean(), mut_samples.mean()])) - (1/50)*logprob_posterior.sum())
    posterior_samples = np.array([np.asarray(fitness_samples), np.asarray(mut_samples)]).T
    waic1, waic2 = WAIC(np.log(kernel(posterior_samples.T)))
    #marginal 95 hdis
    fit_95hdi_low,fit_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=fitness_samples, weights=weights)
    mut_95hdi_low,mut_95hdi_high = pyabc.visualization.credible.compute_credible_interval(vals=mut_samples, weights=weights)
    
    # save relevant values
    f= open(path + out + "_est_real_params.csv","a+")
    f.write(model+',ABC-SMC,' + str(particle_size) + ','.join(str(format(j)) 
                                      for j in
                                      (true_params[0],s_est,true_params[1],
                                       μ_est,s_snv,m_snv,map_dist,
                                       fit_95hdi_low,fit_95hdi_high,
                                       mut_95hdi_low,mut_95hdi_high,
                                       aic, dic,waic1,waic2)) + '\n')
    f.close()
    
    ###PLOTS###
    fig, axes = plt.subplots(4,2)
    
    # text description
    txt = 'ABC-SMC\nModel: ' + model + '\nSimulation id: ' + str(i) + '\nlog10(CNV fitness effect): ' + str(true_params[0]) + '\nlog10(CNV mutation rate): ' + str(true_params[1]) + '\nSNV fitness:' + str(s_snv) + '\nSNV mutation rate:' + str(m_snv) + '\nStarting particle size: ' + str(particle_size)
    
    axes[0,0].axis('off')
    axes[0,0].annotate(txt, (0.1, 0.5), xycoords='axes fraction', va='center')
    
    #observed
    axes[0,1].plot(gens,observation.reshape(-1),linewidth=4)
    axes[0,1].set(ylabel='Proportion of pop with CNV', xlabel='Generation')
    axes[0,1].set_title('Observed data')
    
    # sample size and epsilon
    pyabc.visualization.plot_effective_sample_sizes(history, ax=axes[1,0])
    pyabc.visualization.plot_epsilons(history, ax=axes[1,1])
    
    #joint
    pyabc.visualization.plot_kde_2d_highlevel(history, 'm', 's', numx=100, numy=100, ax=axes[3,0])
    CS = axes[3,0].contour(μ_range, s_range, density, levels = np.quantile(density, q=[0.5,0.95,0.99]),
                colors=('w',),linestyles=('-',),linewidths=(2,))
    fmt = {}
    strs = [ '50%', '95%', '99%']
    for l,s in zip( CS.levels, strs ):
        fmt[l] = s
    axes[3,0].clabel(CS, CS.levels, fmt=fmt, inline=1, fontsize=15)
    axes[3,0].plot(true_params[1],true_params[0], color=red, marker='o', label="simulation parameter")
    axes[3,0].plot(μ_est, s_est, color="k", marker='o', label="MAP estimate")
    axes[3,0].legend(loc='lower left', prop={'size': 5})
    axes[3,0].set(xlabel='log10(CNV mutation rate)', ylabel='log10(CNV fitness effect)')
    
    # marginal (history)
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0,t=t)
        pyabc.visualization.plot_kde_1d(df, w, xmin=np.log10(1e-4), xmax=np.log10(0.3),
                                       x='s', ax=axes[3,1],
                                       label="PDF t={}".format(t))
        pyabc.visualization.plot_kde_1d(df, w, xmin=np.log10(1e-12), xmax=np.log10(1e-3),
                                       x='m', ax=axes[2,0],
                                       label="PDF t={}".format(t))
    axes[3,1].axvline(true_params[0], linestyle='dashed', label="True value", color=red)
    axes[2,0].axvline(true_params[1], linestyle='dashed', label="True value", color=red)
    axes[3,1].axvline(s_est, color="k", label="MAP estimate")
    axes[2,0].axvline(μ_est, color="k", label="MAP estimate")
    axes[3,1].set(xlabel='log10(CNV fitness effect)')
    axes[2,0].set(xlabel='log10(mutation rate)')
    axes[3,1].legend(prop={'size': 5})
    axes[2,0].legend(prop={'size': 5})
    
    #posterior prediction
    for j in range(49) :
        axes[2,1].plot(gens, evo_reps_posterior[j,:], color='blue',alpha=0.1)
    axes[2,1].plot(gens, evo_reps_posterior[49,:], color='blue',alpha=0.1, 
                   label = 'posterior prediction')
    axes[2,1].plot(gens,observation.reshape(-1),linewidth=4, label = 'observation', color = 'black')
    axes[2,1].plot(gens,post_prediction_map.reshape(-1),linewidth=4, alpha=0.8, 
                   label = 'MAP posterior prediction', color='blue')
    axes[2,1].legend(loc='lower right')
    axes[2,1].set(xlabel='Generation',ylabel='Proportion of population with CNV')
    
    plt.gcf().set_size_inches((12, 12))
    plt.gcf().tight_layout()
    plt.title('')
    sns.despine()
    plt.savefig(path + str(i) + out + '.pdf', bbox_inches="tight")
    
x = [path + a for a in os.listdir(path) if a.endswith(out + ".pdf")]
merger = PdfFileMerger()
for pdf in x:
    merger.append(open(pdf, 'rb'))
with open(path + out + "_all.pdf", "wb") as fout:
    merger.write(fout)


    
    