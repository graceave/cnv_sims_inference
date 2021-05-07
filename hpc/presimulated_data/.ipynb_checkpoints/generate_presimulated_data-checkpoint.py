# -*- coding: utf-8 -*-
"""
Created on Feb 18 10:08:39 2021
cnv_sim_sbi

@author: grace
"""
import random
import argparse

import numpy as np
from numpy.random import normal
import os

from cnv_simulation import CNVsimulator_simpleWF, CNVsimulator_simpleChemo

import sbi.utils as utils
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-n', "--number")
args = parser.parse_args()
num = int(args.number)


#####other parameters needed for model #####
# pop size, fitness SNVs, mutation rate SNVs, number of generations
N = 3e8
s_snv=0.001 
m_snv=1e-5 
reps=1
generation = np.array(range(0,268))
# chemostat parameters
A_inoculation = 1e5
S_init = .800
D=0.12
μA=0.45
k=.103
y=32445000
S0=.800
τ=1/10

#### prior ####
prior_min = np.log10(np.array([1e-4,1e-12]))
prior_max = np.log10(np.array([0.3,1e-3]))
prior = utils.BoxUniform(low=torch.tensor(prior_min), 
                         high=torch.tensor(prior_max))

EvoModel = "Chemo"
#### sbi simulator ####
def CNVsimulator(cnv_params):
    cnv_params = np.asarray(torch.squeeze(cnv_params,0))
    reps = 1
    if EvoModel == "WF":
        states = CNVsimulator_simpleWF(reps = reps, N=N, s_snv=s_snv, m_snv=m_snv, generation=generation, seed=None, parameters=cnv_params)
    if EvoModel == "Chemo":
        states = CNVsimulator_simpleChemo(reps = reps, A_inoculation=A_inoculation, S_init=S_init, k=k, D=D, μA=μA, m_snv=m_snv, s_snv=s_snv, S0=S0, y=y, τ=τ, seed=None, parameters=cnv_params)
        
    return states


simulator, prior = prepare_for_sbi(CNVsimulator, prior)

theta_presimulated, x_presimulated = simulate_for_sbi(simulator, proposal=prior, num_simulations=1000, num_workers=1)

#save presimulated thetas and data to csvs
np.savetxt("Chemo_presimulated_theta_1000_" + str(num) +".csv", theta_presimulated.numpy(), delimiter=',')
np.savetxt("Chemo_presimulated_data_1000_" + str(num) +".csv", x_presimulated.numpy(), delimiter=',')
