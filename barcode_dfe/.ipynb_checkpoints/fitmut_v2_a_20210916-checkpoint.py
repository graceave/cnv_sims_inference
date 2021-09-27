import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import Bounds
import math
from tqdm import tqdm
import argparse
import itertools
import csv

read_num_t1_global = None
read_depth_seq_t1t2_global = None
cell_depth_seq_t1t2_global = None
t_seq_t1t2_global = None
PdfConditional_measure_global = None
read_num_t2_right_global = None

read_num_measure_global = None
cell_num_measure_global = None
read_depth_seq_global = None
cell_depth_seq_global = None
x_mean_global = None
t_seq_global = None
seq_num_global = None
kappa_seq_global = None
c_global = None
Ub_global = None

############################################################
def fun_likelihood_lineage_adp_opt(x):
    """ 
    --------------------------------------------------------
    A SUB-FUNCTION CALLED BY MAIN FUNCTION main() TO 
    CALCULATE THE SUM OF THE NEGATIVE LOG LIKELIHOOD VALUE 
    OF EACH LINEAGE
    
    INPUTS
    -x: the fitness and the establishment time of the mutation, 
        [s, tau]
    
    OUTPUTS
    -likelihood_log: negative log-likelihood value for lineage
    --------------------------------------------------------
    """
    global read_num_measure_global  # measured number of reads
    global cell_num_measure_global  # measured number of cells
    global read_depth_seq_global
    global cell_depth_seq_global
    global x_mean_global
    global t_seq_global
    global seq_num_global
    global kappa_seq_global
    global c_global
    global Ub_global

    s = x[0]
    tau = x[1]
    tau = tau*100
                                             
    tempt_establish = 0.01 #0.005
    x_mean_tau = np.interp(tau, t_seq_global, x_mean_global)
    established_size_cell_num = (2*c_global / (s - x_mean_tau) * ((s - x_mean_tau) >= tempt_establish)
                                     + 2*c_global / tempt_establish * ((s - x_mean_tau) < tempt_establish))
    
    cell_num_theory_adaptive_mutant = 1e-1*np.ones(seq_num_global, dtype=float)
    
    if t_seq_global[0] < tau < t_seq_global[-1]:
        pos_index = [[k, k+1] for k in range(len(t_seq_global)-1) if t_seq_global[k] < tau <= t_seq_global[k + 1]][0]
        
        if x_mean_global[pos_index[1]] != x_mean_tau:
            cell_num_theory_adaptive_mutant[pos_index[1]] = np.multiply(established_size_cell_num, 
                                                                        np.exp((t_seq_global[pos_index[1]] - tau) 
                                                                               * (np.log(1 + s) + 1) 
                                                                               - (t_seq_global[pos_index[1]] - tau)
                                                                               /(x_mean_global[pos_index[1]] - x_mean_tau)
                                                                               * ((x_mean_global[pos_index[1]] + 1) 
                                                                                  * np.log(x_mean_global[pos_index[1]] + 1)
                                                                                  - (x_mean_tau + 1) 
                                                                                  * np.log(x_mean_tau + 1))))   
        else:
            cell_num_theory_adaptive_mutant[pos_index[1]] = np.multiply(established_size_cell_num,
                                                                        np.exp((t_seq_global[pos_index[1]] - tau)
                                                                               * np.log((1 + s) / (1 + x_mean_tau))))
            
        if pos_index[1] + 1 < seq_num_global:
            for k in range(pos_index[1] + 1, seq_num_global):
                if x_mean_global[k] != x_mean_global[k - 1]:
                    cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1], 
                                                                     np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                            * (np.log(1 + s) + 1) 
                                                                            - (t_seq_global[k] - t_seq_global[k - 1])
                                                                            /(x_mean_global[k] - x_mean_global[k - 1])
                                                                            * ((x_mean_global[k] + 1)
                                                                               * np.log(x_mean_global[k] + 1)
                                                                               - (x_mean_global[k - 1] + 1)
                                                                               * np.log(x_mean_global[k - 1] + 1))))
                else:
                    cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1],
                                                                     np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                            * np.log((1 + s) / (1 + x_mean_global[k - 1]))))
                                                                
    elif tau <= t_seq_global[0]:        
        if x_mean_global[0] != x_mean_tau:
            cell_num_theory_adaptive_mutant[0] = np.multiply(established_size_cell_num, 
                                                             np.exp((t_seq_global[0] - tau) 
                                                                    * (np.log(1 + s) + 1)
                                                                    - (t_seq_global[0] - tau)
                                                                    / (x_mean_global[0] - x_mean_tau)
                                                                    * ((x_mean_global[0] + 1) 
                                                                       * np.log(x_mean_global[0] + 1)
                                                                       - (x_mean_tau + 1) 
                                                                       * np.log(x_mean_tau + 1))))
        else:
            cell_num_theory_adaptive_mutant[0] = np.multiply(established_size_cell_num, 
                                                             np.exp((t_seq_global[0] - tau)
                                                                    * np.log((1 + s) / (1 + x_mean_tau))))
        
        for k in range(1, seq_num_global):
            if x_mean_global[k] != x_mean_global[k - 1]:                                                             
                cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1],
                                                                 np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                        * (np.log(1 + s) + 1)
                                                                        - (t_seq_global[k] - t_seq_global[k - 1])
                                                                        /(x_mean_global[k] - x_mean_global[k - 1])
                                                                        * ((x_mean_global[k] + 1)
                                                                           * np.log(x_mean_global[k] + 1)
                                                                           - (x_mean_global[k - 1] + 1)
                                                                           * np.log(x_mean_global[k - 1] + 1)))) 
            else:
                cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1],
                                                                 np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                        *np.log((1 + s) / (1 + x_mean_global[k - 1]))))
    
    # cell_num_theory_adaptive: estimated cell number (both non-mutant and mutant cells) of a adaptive lineage
    cell_num_theory_adaptive = 1e-1*np.ones(seq_num_global, dtype=float)
    cell_num_theory_adaptive[0] = np.max([cell_num_measure_global[0], cell_num_theory_adaptive_mutant[0]])    
    for k in range(1, seq_num_global):
        tempt_nonmutant = np.max([cell_num_theory_adaptive[k - 1] - cell_num_theory_adaptive_mutant[k - 1], 1e-1])
        #tempt_nonmutant = np.max([cell_num_measure_global[k - 1] - cell_num_theory_adaptive_mutant[k - 1], 1e-1])
        if x_mean_global[k] != x_mean_global[k - 1]:
            cell_num_theory_adaptive[k] = (cell_num_theory_adaptive_mutant[k] 
                                           + np.multiply(tempt_nonmutant, 
                                                         np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                - (t_seq_global[k] - t_seq_global[k - 1])
                                                                / (x_mean_global[k] - x_mean_global[k - 1])
                                                                * ((x_mean_global[k] + 1) 
                                                                   * np.log(x_mean_global[k] + 1)
                                                                   - (x_mean_global[k-1] + 1) 
                                                                   * np.log(x_mean_global[k-1] + 1)))))
        else:
            cell_num_theory_adaptive[k] = (cell_num_theory_adaptive_mutant[k] 
                                           + np.multiply(tempt_nonmutant, 
                                                         np.exp((t_seq_global[k] - t_seq_global[k - 1]) 
                                                                * np.log(1 / (1 + x_mean_global[k - 1])))))
                                                                           
    ratio = np.true_divide(read_depth_seq_global, cell_depth_seq_global)
    read_num_theory_adaptive = np.multiply(cell_num_theory_adaptive, ratio)
    
    read_num_theory_adaptive[read_num_theory_adaptive<=1] = 1e-1

    # Calculate likelihood (in log)
    likelihood_adaptive_log = (0.25 * np.log(read_num_theory_adaptive)
                                            - 0.5 * np.log(4 * np.pi * kappa_seq_global)
                                            - 0.75 * np.log(read_num_measure_global)
                                            - (np.sqrt(read_num_measure_global)
                                               - np.sqrt(read_num_theory_adaptive)) ** 2 / kappa_seq_global)
     
     
    delta_s = 0.005
    tempt1 = Ub_global * np.exp(-s / 0.1) / 0.1 * delta_s
    ##########
    #probability_prior_adaptive_log = (np.log(tempt1) 
    #                                  + np.log(s / sp.special.gamma(cell_num_measure_global[0] * tempt1)/c_global)
    #                                  - cell_num_measure_global[0] * tempt1 * s / c_global * tau 
    #                                  - np.exp(-s * tau))  # need change
    
    # defined in Jamie's original code
    #probability_prior_adaptive_log = np.log(np.max([s, 1e-8]) * cell_num_measure_global[0] * tempt1)
    
    #probability_prior_adaptive_log = np.log(np.max([s, 1e-8]) * tempt1)
    probability_prior_adaptive_log = np.log(tempt1*1/250)
    ##########
    
    likelihood_log = np.sum(likelihood_adaptive_log) + probability_prior_adaptive_log 

    return -likelihood_log



############################################################
def fun_likelihood_lineage_est(x):
    """ 
    --------------------------------------------------------
    A SUB-FUNCTION CALLED BY MAIN FUNCTION main() TO 
    CALCULATE THE SUM OF THE NEGATIVE LOG LIKELIHOOD VALUE 
    OF EACH LINEAGE
    
    INPUTS
    -x: the fitness and the establishment time of the mutation, 
        [s, tau]
    
    OUTPUTS
    -
    --------------------------------------------------------
    """
    global read_num_measure_global  # measured number of reads
    global cell_num_measure_global  # measured number of cells
    global read_depth_seq_global
    global cell_depth_seq_global
    global x_mean_global
    global t_seq_global
    global seq_num_global
    global kappa_seq_global
    global c_global
    global Ub_global

    s = x[0]
    tau = x[1]
    tau = tau*100
                                             
    tempt_establish = 0.01 #0.005
    x_mean_tau = np.interp(tau, t_seq_global, x_mean_global)
    established_size_cell_num = (2*c_global / (s - x_mean_tau) * ((s - x_mean_tau) >= tempt_establish)
                                     + 2*c_global / tempt_establish * ((s - x_mean_tau) < tempt_establish))
    
    cell_num_theory_adaptive_mutant = 1e-1*np.ones(seq_num_global, dtype=float)
    
    if t_seq_global[0] < tau < t_seq_global[-1]:
        pos_index = [[k, k+1] for k in range(len(t_seq_global)-1) if t_seq_global[k] < tau <= t_seq_global[k + 1]][0]
        
        if x_mean_global[pos_index[1]] != x_mean_tau:
            cell_num_theory_adaptive_mutant[pos_index[1]] = np.multiply(established_size_cell_num, 
                                                                        np.exp((t_seq_global[pos_index[1]] - tau) 
                                                                               * (np.log(1 + s) + 1) 
                                                                               - (t_seq_global[pos_index[1]] - tau)
                                                                               /(x_mean_global[pos_index[1]] - x_mean_tau)
                                                                               * ((x_mean_global[pos_index[1]] + 1) 
                                                                                  * np.log(x_mean_global[pos_index[1]] + 1)
                                                                                  - (x_mean_tau + 1) 
                                                                                  * np.log(x_mean_tau + 1))))   
        else:
            cell_num_theory_adaptive_mutant[pos_index[1]] = np.multiply(established_size_cell_num,
                                                                        np.exp((t_seq_global[pos_index[1]] - tau)
                                                                               * np.log((1 + s) / (1 + x_mean_tau))))
            
        if pos_index[1] + 1 < seq_num_global:
            for k in range(pos_index[1] + 1, seq_num_global):
                if x_mean_global[k] != x_mean_global[k - 1]:
                    cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1], 
                                                                     np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                            * (np.log(1 + s) + 1) 
                                                                            - (t_seq_global[k] - t_seq_global[k - 1])
                                                                            /(x_mean_global[k] - x_mean_global[k - 1])
                                                                            * ((x_mean_global[k] + 1)
                                                                               * np.log(x_mean_global[k] + 1)
                                                                               - (x_mean_global[k - 1] + 1)
                                                                               * np.log(x_mean_global[k - 1] + 1))))
                else:
                    cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1],
                                                                     np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                            * np.log((1 + s) / (1 + x_mean_global[k - 1]))))
                                                                
    elif tau <= t_seq_global[0]:        
        if x_mean_global[0] != x_mean_tau:
            cell_num_theory_adaptive_mutant[0] = np.multiply(established_size_cell_num, 
                                                             np.exp((t_seq_global[0] - tau) 
                                                                    * (np.log(1 + s) + 1)
                                                                    - (t_seq_global[0] - tau)
                                                                    / (x_mean_global[0] - x_mean_tau)
                                                                    * ((x_mean_global[0] + 1) 
                                                                       * np.log(x_mean_global[0] + 1)
                                                                       - (x_mean_tau + 1) 
                                                                       * np.log(x_mean_tau + 1))))
        else:
            cell_num_theory_adaptive_mutant[0] = np.multiply(established_size_cell_num, 
                                                             np.exp((t_seq_global[0] - tau)
                                                                    * np.log((1 + s) / (1 + x_mean_tau))))
        
        for k in range(1, seq_num_global):
            if x_mean_global[k] != x_mean_global[k - 1]:                                                             
                cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1],
                                                                 np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                        * (np.log(1 + s) + 1)
                                                                        - (t_seq_global[k] - t_seq_global[k - 1])
                                                                        /(x_mean_global[k] - x_mean_global[k - 1])
                                                                        * ((x_mean_global[k] + 1)
                                                                           * np.log(x_mean_global[k] + 1)
                                                                           - (x_mean_global[k - 1] + 1)
                                                                           * np.log(x_mean_global[k - 1] + 1)))) 
            else:
                cell_num_theory_adaptive_mutant[k] = np.multiply(cell_num_theory_adaptive_mutant[k - 1],
                                                                 np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                        *np.log((1 + s) / (1 + x_mean_global[k - 1]))))
    
    # cell_num_theory_adaptive: estimated cell number (both non-mutant and mutant cells) of a adaptive lineage
    cell_num_theory_adaptive = 1e-1*np.ones(seq_num_global, dtype=float)
    cell_num_theory_adaptive[0] = np.max([cell_num_measure_global[0], cell_num_theory_adaptive_mutant[0]])    
    for k in range(1, seq_num_global):
        tempt_nonmutant = np.max([cell_num_theory_adaptive[k - 1] - cell_num_theory_adaptive_mutant[k - 1], 1e-1])
        #tempt_nonmutant = np.max([cell_num_measure_global[k - 1] - cell_num_theory_adaptive_mutant[k - 1], 1e-1])
        if x_mean_global[k] != x_mean_global[k - 1]:
            cell_num_theory_adaptive[k] = (cell_num_theory_adaptive_mutant[k] 
                                           + np.multiply(tempt_nonmutant, 
                                                         np.exp((t_seq_global[k] - t_seq_global[k - 1])
                                                                - (t_seq_global[k] - t_seq_global[k - 1])
                                                                / (x_mean_global[k] - x_mean_global[k - 1])
                                                                * ((x_mean_global[k] + 1) 
                                                                   * np.log(x_mean_global[k] + 1)
                                                                   - (x_mean_global[k-1] + 1) 
                                                                   * np.log(x_mean_global[k-1] + 1)))))
        else:
            cell_num_theory_adaptive[k] = (cell_num_theory_adaptive_mutant[k] 
                                           + np.multiply(tempt_nonmutant, 
                                                         np.exp((t_seq_global[k] - t_seq_global[k - 1]) 
                                                                * np.log(1 / (1 + x_mean_global[k - 1])))))
                                                                           
    ratio = np.true_divide(read_depth_seq_global, cell_depth_seq_global)
    read_num_theory_adaptive = np.multiply(cell_num_theory_adaptive, ratio)
    
    read_num_theory_adaptive[read_num_theory_adaptive<=1] = 1e-1

    # Calculate likelihood (in log)
    likelihood_adaptive_log = (0.25 * np.log(read_num_theory_adaptive)
                                            - 0.5 * np.log(4 * np.pi * kappa_seq_global)
                                            - 0.75 * np.log(read_num_measure_global)
                                            - (np.sqrt(read_num_measure_global)
                                               - np.sqrt(read_num_theory_adaptive)) ** 2 / kappa_seq_global)
     
     
    delta_s = 0.005
    tempt1 = Ub_global * np.exp(-s / 0.1) / 0.1 * delta_s
    ##########
    #probability_prior_adaptive_log = (np.log(tempt1) 
    #                                  + np.log(s / sp.special.gamma(cell_num_measure_global[0] * tempt1)/c_global)
    #                                  - cell_num_measure_global[0] * tempt1 * s / c_global * tau 
    #                                  - np.exp(-s * tau))  # need change
    
    # defined in Jamie's original code
    #probability_prior_adaptive_log = np.log(np.max([s, 1e-8]) * cell_num_measure_global[0] * tempt1)
    
    #probability_prior_adaptive_log = np.log(np.max([s, 1e-8]) * tempt1)
    probability_prior_adaptive_log = np.log(tempt1*1/250)
    ##########
    
    likelihood_log_adp = np.sum(likelihood_adaptive_log) + probability_prior_adaptive_log 
    
    likelihood_log_neu = np.sum(likelihood_adaptive_log)

    return likelihood_log_adp, likelihood_log_neu, cell_num_theory_adaptive_mutant, cell_num_theory_adaptive



############################################################
def main():
    """ 
    --------------------------------------------------------
    ESTIMATE FITNESS AND ESTABLISHMENT TIME OF EACH SPONTANEOUS 
    ADAPTIVE MUTATION IN COMPETITIVE POOLED GROWTH OF A ISOGENIC 
    POPULATION
    
    OPTIONS
    --input: a .csv file, with each column being the read number 
             per lineage at each sequenced time-point
    --t_seq: a .csv file, with a column of sequenced time-points 
             in number of generations
    --output_filename: prefix of output .csv files (default: output)
    
    OUTPUTS
    output_filename_FitMut_Result.csv: 
      1st column: estimated fitness of each lineage, [x1, x2, ...],
      2nd column: log likelihood value of each lineage, [f1, f2, ...],
      3rd column: estimated mean fitness per sequenced time-point 
                  [x_mean(0), x_mean(t1), ...],
      4th column+: estimated reads number per genotype per 
                   sequencing time-point, with each time-point 
                   being a column
    --------------------------------------------------------               
    """
    global read_num_t1_global
    global read_depth_seq_t1t2_global
    global cell_depth_seq_t1t2_global
    global t_seq_t1t2_global
    global PdfConditional_measure_global
    global read_num_t2_right_global

    global read_num_measure_global
    global cell_num_measure_global
    global read_depth_seq_global
    global cell_depth_seq_global
    global x_mean_global
    global t_seq_global
    global seq_num_global
    global kappa_seq_global
    global c_global
    global Ub_global

    parser = argparse.ArgumentParser(description='Estimate fitness and establishment time of each spontanuous adaptive '
                                                 'mutations in a competitive pooled growth experiment',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', type=str, help='a .csv file: with each column being the read number per '
                                                        'genotype at each sequenced time-point')
    parser.add_argument('-t', '--t_seq', type=str, 
                        help='a .csv file of 2 columns:'
                             '1st column is sequenced time-points evaluated in number of generations, '
                             '2nd column is total effective number of cells of the population for each sequenced time-point.')
    parser.add_argument('-u', '--mutation_rate', type=float, default=1e-5, help='total beneficial mutation rate')
    parser.add_argument('-c', '--c', type=float, default=0.5, help='a noise parameter that characterizes the cell '
                                                                   'growth and cell transfer')  # might need change
    parser.add_argument('-o', '--output_filename', type=str, default='output', help='prefix of output .csv files')

    args = parser.parse_args()
    read_num_seq = np.array(pd.read_csv(args.input, header=None), dtype=float)
    read_num_seq[read_num_seq == 0] = 1e-1
    read_depth_seq_global = np.sum(read_num_seq, axis=0)
    
    lineages_num, seq_num_global = np.shape(read_num_seq)   
    
    csv_input = pd.read_csv(args.t_seq, header=None)
    t_seq_global = np.array(csv_input[0][~pd.isnull(csv_input[0])], dtype=float)
    cell_depth_seq_global = np.array(csv_input[1][~pd.isnull(csv_input[1])], dtype=float)
    
    cell_num_seq = read_num_seq / read_depth_seq_global * cell_depth_seq_global
    
    Ub_global = args.mutation_rate
    c_global = args.c
    output_filename = args.output_filename

    ############################## 
    kappa_seq_global = np.zeros(seq_num_global, dtype=float)
    kappa_seq_global[0] = 2.5
    
    for k in range(seq_num_global-1):
        read_num_t1_left, read_num_t1_right = 20, 40
        read_num_t2_left, read_num_t2_right = 0, 4*read_num_t1_right
    
        kappa = np.zeros(read_num_t1_right - read_num_t1_left, dtype=float)
        
        read_depth_seq_t1t2_global = read_depth_seq_global[k:k + 2]
        t_seq_t1t2_global = t_seq_global[k:k + 2]
        read_num_t2_right_global = read_num_t2_right
        
        for read_num_t1 in range(read_num_t1_left, read_num_t1_right):
            read_num_t1_global = read_num_t1
            pos = read_num_seq[:, k] == read_num_t1
            
            if np.sum(pos)>100:
                PdfConditional_measure_global = np.histogram(read_num_seq[pos, k + 1],
                                                             bins=np.arange(read_num_t2_left, read_num_t2_right + 0.001),
                                                             density=True)[0]
        
                dist_x = np.arange(read_num_t2_left, read_num_t2_right)
                param_a = np.sum(dist_x*PdfConditional_measure_global)
                param_b = np.sum((dist_x - param_a)**2*PdfConditional_measure_global)/(2*param_a)

                kappa[read_num_t1 - read_num_t1_left] = param_b
                
            pos = kappa>0
            if np.sum(pos)>0:
                kappa_mean = np.mean(kappa[kappa>0])
                kappa_seq_global[k+1] = np.mean(kappa[pos])
            
    pos_1 = kappa_seq_global==0
    if np.sum(pos_1)>0:
        pos_2 = kappa_seq_global!=0
        kappa_seq_global[pos_1] = np.mean(kappa_seq_global[pos_2])
    ##########
    
    
    
   
    # estimate fitness and establishment time of each adaptive mutation
    x_mean_seq_dict = dict()
    mutant_fraction_dict = dict()
    
    iter_num = 30
    x0 = [0.01, 50/100]
    bounds = Bounds([0.001, -1000/100], [0.5, math.floor(t_seq_global[-1] - 1)/100])
    for k in tqdm(range(1, iter_num+1)):
        if k == 1:
            x_mean_global = np.zeros(seq_num_global)
        else:
            x_mean_global = x_mean_seq_dict[k-1]
            
        result_x = np.zeros(lineages_num)
        result_tau = np.zeros(lineages_num)
        result_likelihood_log_adp = np.zeros(lineages_num)
        result_likelihood_log_neu = np.zeros(lineages_num)
        
        mutant_fraction_numerator = np.zeros(seq_num_global)
        x_mean_numerator = np.zeros(seq_num_global)
        cell_depth_seq_theory = np.zeros(seq_num_global)
        
        ###
        cell_num_seq_mutant = np.zeros(np.shape(read_num_seq))
        ###
        
        for i in range(int(lineages_num)):
            read_num_measure_global = read_num_seq[i, :]
            cell_num_measure_global = cell_num_seq[i, :]
            
            if k==1:
                x0 = [0.01, 50/100]
            else:
                x0 = [result_x[i], result_tau[i]/100]
            
            opt_output = minimize(fun_likelihood_lineage_adp_opt, x0, method='L-BFGS-B', bounds=bounds,
                                  options={'ftol': 1e-8, 'gtol': 1e-8, 
                                           'eps':1e-8, 'maxls':100, 'disp': False})
            
            result_x[i], result_tau[i] = opt_output.x[0], opt_output.x[1]*100
            result_likelihood_log_adp[i] = -opt_output.fun
            
            _, result_likelihood_log_neu[i], _, _ = fun_likelihood_lineage_est([1e-5, 0])                     
            
            ##########
            cell_depth_seq_theory += cell_num_measure_global
            
            if result_likelihood_log_adp[i] - result_likelihood_log_neu[i] > 0:
                _, _, cell_num_theory_adaptive_mutant, cell_num_theory_adaptive = fun_likelihood_lineage_est(opt_output.x)
                x_mean_numerator += cell_num_theory_adaptive_mutant * opt_output.x[0]
                mutant_fraction_numerator += cell_num_theory_adaptive_mutant
                #cell_depth_seq_theory += cell_num_theory_adaptive
                ###
                cell_num_seq_mutant[i,:] = cell_num_theory_adaptive_mutant
                ###
            #else:
            #    _, _, _, cell_num_theory_neutral = fun_likelihood_lineage_est([1e-5, 0])
                #cell_depth_seq_theory += cell_num_theory_neutral
                
        x_mean_seq_dict[k] = x_mean_numerator/cell_depth_seq_theory
        mutant_fraction_dict[k] = mutant_fraction_numerator/cell_depth_seq_theory
        
       
    result_likelihood_log = result_likelihood_log_adp - result_likelihood_log_neu
    idx_adaptive_inferred = np.where(result_likelihood_log > 0)[0]
    idx_neutral_inferred = np.where(result_likelihood_log <= 0)[0]
    result_x[idx_neutral_inferred] = 0
    result_tau[idx_neutral_inferred] = 0
    
    result_output = {'Mutation_Fitness': result_x,
                     'Establishment_Time': result_tau,
                     'Likelihood_Log': result_likelihood_log,
                     'Likelihood_Log_Adaptive': result_likelihood_log_adp,
                     'Likelihood_Log_Neutral': result_likelihood_log_neu,
                     'Mean_Fitness': x_mean_seq_dict[iter_num],
                     'Kappa_Value':kappa_seq_global, 
                     'Mutant_Cell_Fraction': mutant_fraction_dict[iter_num]}
    
    tempt = list(itertools.zip_longest(*list(result_output.values())))
    with open(output_filename + '_MutSeq_Result.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(result_output.keys())
        w.writerows(tempt)
        
    tempt = list(itertools.zip_longest(*list(x_mean_seq_dict.values())))
    with open(output_filename + '_Mean_fitness_Result.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(x_mean_seq_dict.keys())
        w.writerows(tempt)
        
    ###
    tempt = pd.DataFrame(cell_num_seq_mutant, dtype = int)
    tempt.to_csv(output_filename + '_Cell_Number_Mutant_Estimated.csv', index=False, header=False)
    
    tempt = pd.DataFrame(cell_num_seq, dtype = int)
    tempt.to_csv(output_filename + '_Cell_Number.csv', index=False, header=False)
    ###
            
if __name__ == "__main__":
    main()
