import numpy as np
import numba

## Wright Fisher stochastic discrete time model ##
def CNVsimulator_simpleWF(N, s_snv, m_snv, generation, seed=None, **kwargs):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for 267 generations
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array of length 25
    
    Parameters
    -------------------
    N : int
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations 0 through the end of simulation
    seed : int
    
    Depending on what the downstream inference is, the following parameters are can be passed
    cnv_params : np.array, 1d of length dim_param
        Parameter vector with the log 10 selection coefficient and log 10 cnv mutation rate, for use with SNPE or to build observed data
    parameters : instance of parameters
        has attribute s, float, log 10 selection coefficient
        has attribute m, float, log 10 cnv mutation rate
        for use with pyABC
    """
    cnv_params = kwargs.get('cnv_params', None)
    parameters = kwargs.get('parameters', None)
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)
    if isinstance(cnv_params, np.ndarray):
        s_cnv, m_cnv = np.power(10,cnv_params)
    else:
        s_cnv = np.power(10,parameters.s)
        m_cnv = np.power(10,parameters.m)
    
    w = np.array([1, 1 + s_cnv, 1 + s_snv])
    S = np.diag(w)
    
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
        p_cnv.append(p[1])  # frequency of CNVs
        p = E @ p.reshape((3, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, p) # random genetic drift
    
    #these were the generations observed in Lauer et al. 2018, so the ones we will use here
    exp_gen = np.array([25,33,41,54,62,70,79,87,95,103,116,124,132,145,153,161,174,182,190,211,219,232,244,257,267])
    
    return np.transpose(p_cnv)[exp_gen]


## Chemostat stochastic continuous time model ##

@numba.jit
def get_rates(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv):
    """ Rates
    A :  number of ancestral cells at current time step
    CNV : number of cells with CNV at current time step
    SNV : number of cells with SNV at current time step
    S : substrate concentration at current time step
    k : substrate concentration at half-maximal growth rate (here, the same for all cells)
    D : dilution rate
    μA : ancestral maximum growth rate
    m_cnv : ancestral -> CNV mutation rate
    m_snv : ancestral -> SNV mutation rate
    s_cnv : CNV selection coefficient
    s_snv : SNV selection coefficient
    """
    return np.array([
        A* ((μA *S) / (k + S)), # growth ancestral
        A*D, # dilution ancestral
        CNV * (((μA*(1+s_cnv)) *S) / (k + S)), # growth CNV 
        CNV*D, # dilution CNV
        SNV * (((μA*(1+s_snv)) *S) / (k + S)), # growth SNV
        SNV*D, # dilution SNV
        A * m_cnv, # ancestral -> CNV
        A * m_snv] # ancestral -> SNV
    )

def tau_leap(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv, updates, I, y, τ):
    """ Single update of simulation using tau leaps
    A :  number of ancestral cells at current time step
    CNV : number of cells with CNV at current time step
    SNV : number of cells with SNV at current time step
    S : substrate concentration at current time step
    k : substrate concentration at half-maximal growth rate (here, the same for all cell types)
    D : dilution rate
    μA : ancestral maximum growth rate
    m_cnv : ancestral -> CNV mutation rate
    m_snv : ancestral -> SNV mutation rate
    s_cnv : CNV selection coefficient
    s_snv : SNV selection coefficient
    updates : change in reactants due to reactions
    I : incoming substrate concentration - S0 in diff eq
    y : number of cells produced per mole of the limiting nutrient (here, the same for all cell types)
    τ : amount to advance time
    """
    rates = get_rates(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv)
    j = np.random.poisson(rates * τ)
    if (j > 150).any() and τ > 1/100:
            τ /= 2
    A, CNV, SNV = np.array([A, CNV, SNV]) + np.transpose(updates) @ j
    ΔS = I*D - D*S - (A*μA*S)/((S+k)*y) - (CNV*(μA*(1+s_cnv))*S)/((S+k)*y) - (SNV*(μA*(1+s_snv))*S)/((S+k)*y)
    return τ, A, CNV, SNV, S+ΔS

def CNVsimulator_simpleChemo(A_inoc, S_init, k, D, μA, m_snv, s_snv, I, y, τ, seed=None, **kwargs):
    """ Simulates CNV and SNV evolution in a steady state chemostat for 267 generations, which is 1548.6 hours
    Begins counting generations after 48 hours, during which time the chemostat reaches steady state
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array of length 25
    
    A_inoc :  cell density at inoculation with ancestral type
    S_init : initial substrate concentration in chemostat
    k : substrate concentration at half-maximal growth rate (here, the same for all cell types)
    D : dilution rate
    μA : ancestral maximum growth rate
    m_snv : ancestral -> SNV mutation rate
    s_snv : SNV selection coefficient
    updates : change in reactants due to reactions
    I : incoming substrate concentration - S0 in diff eq
    y : number of cells produced per mole of the limiting nutrient (here, the same for all cell types)
    τ : amount to advance time
    m_cnv : ancestral -> CNV mutation rate
    s_cnv : CNV selection coefficient
    seed : int
    
    Depending on what the downstream inference is, the following parameters are can be passed
    cnv_params : np.array, 1d of length dim_param
        Parameter vector with the log 10 selection coefficient and log 10 cnv mutation rate, for use with SNPE or to build observed data
    parameters : instance of parameters
        has attribute s, float, log 10 selection coefficient
        has attribute m, float, log 10 cnv mutation rate
        for use with pyABC
    """
    cnv_params = kwargs.get('cnv_params', None)
    parameters = kwargs.get('parameters', None)
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert A_inoc > 0
    A_inoc = np.uint64(A_inoc)
    if isinstance(cnv_params, np.ndarray):
        s_cnv, m_cnv = np.power(10,cnv_params)
    else:
        s_cnv = np.power(10,parameters.s)
        m_cnv = np.power(10,parameters.m)
    s_cnv /= 5.8 #convert from per generation to per hour
    m_cnv /= 5.8 #convert from per generation to per hour
    s_snv /= 5.8
    m_snv /= 5.8


    updates = np.array([
        [1, 0, 0], # growth ancestral
        [-1, 0, 0], # dilution ancestral
        [0, 1, 0],  # growth CNV
        [0, -1, 0], # dilution CNV
        [0, 0, 1],  # growth SNV
        [0, 0, -1], # dilution SNV
        [-1, 1, 0], # mutation ancestral -> CNV
        [-1, 0, 1] # mutation ancestral -> SNV
    ])
    A = A_inoc
    S = S_init # initial substrate concentration = concentration in media
    CNV, SNV = 0, 0
    t=0
    while t < 48: # allow chemostat to reach steady state
        τ, A, CNV, SNV, S = tau_leap(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv, updates, I, y, τ)
        t+=τ
        
    t=0
    states = [np.array([A, CNV, SNV, S])]
    times=[t]
    while t < 1548.6: # record from when the chemostat reaches steady state to gen 267
        τ, A, CNV, SNV, S = tau_leap(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv, updates, I, y, τ)
        t+=τ
        states.append(np.array([A, CNV, SNV, S]))
        times.append(t)
    
    data = np.stack(states, axis=1)
    times=np.array(times)
    gens=times/5.8 #convert hours to generations
    #these were the generations observed in Lauer et al. 2018, so the time points at which we will record the state
    exp_gen = np.array([25,33,41,54,62,70,79,87,95,103,116,124,132,145,153,161,174,182,190,211,219,232,244,257,267])
    #get the indices closest to the generations observed
    indices = np.empty(exp_gen.shape, dtype=int)
    for i in range(len(exp_gen)):
        ab = np.abs(times-exp_gen[i])
        indices[i] = ab.argmin()
    # convert pop densities for each of three states to cnv proportion
    p_cnv = data[1,:] / data[0:3,:].sum(axis=0)
    
    return np.transpose(p_cnv)[indices]
