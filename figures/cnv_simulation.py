import numpy as np
import numba

def get_params(parameters):
    if isinstance(parameters, np.ndarray):
        s_cnv, m_cnv = np.power(10,parameters)
    if not isinstance(parameters, np.ndarray):
        s_cnv = np.power(10,parameters.s)
        m_cnv = np.power(10,parameters.m)
    return s_cnv,m_cnv

def get_params_multi(parameters):
    if isinstance(parameters, np.ndarray):
        shape = parameters[0]
        scale, m_cnv = np.power(10,parameters[1:])
    if not isinstance(parameters, np.ndarray):
        shape = parameters.gamma_shape
        scale = np.power(10,parameters.gamma_scale)
        m_cnv = np.power(10,parameters.m)
    return shape,scale,m_cnv

## Wright Fisher stochastic discrete time model ##
def simpleWF(N, s_snv, m_snv, generation, s_cnv, m_cnv, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for x generations
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array same length as generation
    
    Parameters
    -------------------
    N : int
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
    
    s_cnv : float
        fitness benefit of CNVs  
    m_cnv : float 
        probability mutation to CNV 
    """
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)
    
    w = np.array([1, 1 + s_cnv, 1 + s_snv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cnv - m_snv, 0, 0],
                [m_cnv, 1, 0],
                [m_snv, 0, 1]], dtype='float64')
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
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_cnv.append(p[1])  # frequency of CNVs
        p = E @ p.reshape((3, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    return np.transpose(p_cnv)[generation.astype(int)]

def CNVsimulator_simpleWF(reps, N, s_snv, m_snv, generation, parameters, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for some generations
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array of length len(generation)
    
    Parameters
    -------------------
    reps: int
        number of replicate simulations to perform
    N : int
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
    
     parameters : instance of parameters
        has attribute s, float, log 10 selection coefficient
        has attribute m, float, log 10 cnv mutation rate
        for use with pyABC
        OR
        np.array, 1d of length dim_param
        Parameter vector with the log 10 selection coefficient and log 10 cnv mutation rate, for use with SNPE or to build observed data
    """
    s_cnv, m_cnv = get_params(parameters)
    evo_reps = []
    for i in range(reps):
        out=simpleWF(N, s_snv, m_snv, generation, s_cnv, m_cnv, seed=seed)
        evo_reps.append(out)
    return np.array(evo_reps)

def CNVsimulator_multiWF(reps, N, s_snv, m_snv, generation, parameters, return_s=False, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for some generations
    Returns proportion of the population with a CNV for generation for n_pops populations as 2d np.array of generation x n_pop
    
    Parameters
    -------------------
    reps: int
        number of replicate simulations to perform
    N : int
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
     parameters : instance of parameters
        has attribute gamma_shape, float, parameterizes gamma distribution that CNV selection coefficient is drawn from
        has attribute log10 gamma_scale, float, parameterizes gamma distribution that CNV selection coefficient is drawn from
        has attribute m, float, log 10 cnv mutation rate
        for use with pyABC
        OR
        np.array, 1d of length dim_param
        Parameter vector with the gamma_shape, log10 gamma_scale, and log 10 cnv mutation rate, for use with SNPE or to build observed data
    """
    all_obs = []
    shape, scale, m_cnv = get_params_multi(parameters)
    s_cnv = np.random.gamma(shape=shape, scale=scale, size=reps)
    for i in range(reps):
        obs = simpleWF(N, s_snv, m_snv, generation, s_cnv=s_cnv[i], m_cnv=m_cnv, seed=seed)    
        all_obs.append(obs)
    if return_s == False:
        return np.vstack(all_obs)
    if return_s == True:
        return np.vstack(all_obs),s_cnv

## Chemostat stochastic continuous time model ##
# chemostat parameters
A_inoculation = 1.5e7
S_init = .800
D=0.12
μA=0.35
k=.103
y=32445000*20
S0=.800
τ=1/10

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
        CNV * (((μA+s_cnv) *S) / (k + S)), # growth CNV 
        CNV*D, # dilution CNV
        SNV * (((μA+s_snv) *S) / (k + S)), # growth SNV
        SNV*D, # dilution SNV
        A * m_cnv, # ancestral -> CNV
        A * m_snv],#, # ancestral -> SNV
        dtype='float64'
    )

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

def τ_leap(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv, S0, y, τ):
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
    S0 : incoming substrate concentration - S0 in diff eq
    y : number of cells produced per mole of the limiting nutrient (here, the same for all cell types)
    τ : amount to advance time
    """
    rates = get_rates(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv)
    try:
        adj_rates = np.random.poisson(rates * τ)
    except ValueError:
        print(rates, τ)
        raise
    if (adj_rates > 150).any() and τ > 1/100:
        τ /= 2
    ΔA, ΔCNV, ΔSNV = updates.T @ adj_rates
    A += ΔA
    CNV += ΔCNV
    SNV += ΔSNV
    ΔS = S0*D - D*S - (A*μA*S)/((S+k)*y) - (CNV*(μA+s_cnv)*S)/((S+k)*y) - (SNV*(μA+s_snv)*S)/((S+k)*y)
    return τ, A, CNV, SNV, S+ΔS

def simpleChemo(generation, A_inoculation, S_init, k, D, μA, m_snv, s_snv, S0, y, τ, s_cnv, m_cnv, seed=None):
    """ Simulates CNV and SNV evolution in a steady state chemostat for some generations
    Begins counting generations after 48 hours, during which time the chemostat reaches steady state
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array of length len(generation)
    
    generation : np.array, 1d 
        with generations to output
    A_inoculation :  cell density at inoculation with ancestral type
    S_init : initial substrate concentration in chemostat
    k : substrate concentration at half-maximal growth rate (here, the same for all cell types)
    D : dilution rate
    μA : ancestral maximum growth rate
    m_snv : ancestral -> SNV mutation rate
    s_snv : SNV selection coefficient
    updates : change in reactants due to reactions
    S0 : incoming substrate concentration - S0 in diff eq
    y : number of cells produced per mole of the limiting nutrient (here, the same for all cell types)
    τ : amount to advance time
    m_cnv : ancestral -> CNV mutation rate
    s_cnv : CNV selection coefficient
    seed : int
    """
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert A_inoculation > 0
    A_inoculation = np.uint64(A_inoculation) 
    
    s_cnv = (s_cnv*μA)/np.log(2) #convert from per generation to per unit time
    m_cnv /= 5.8 #convert from per generation to per hour
    s_snv = (s_snv*μA)/np.log(2)
    m_snv /= 5.8


    A = A_inoculation
    S = S_init # initial substrate concentration = concentration in media
    CNV, SNV = 0, 0
    t=0
    
    while t < 48: # allow chemostat to reach steady state
        τ, A, CNV, SNV, S = τ_leap(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv, S0, y, τ)
        A = max(A, 0)
        CNV = max(CNV, 0)
        SNV = max(SNV, 0)
        S = max(S, 0)
        S = min(S, S_init)
        t += τ
        
    t=0
    states = [np.array([A, CNV, SNV, S])]
    times=[t]
    max_t = generation.max()*5.8
    while t < max_t: # record from when the chemostat reaches steady state to hour that corresponds with the last generation
        τ, A, CNV, SNV, S = τ_leap(A, CNV, SNV, S, k, D, μA, m_cnv, m_snv, s_cnv, s_snv, S0, y, τ)
        A = max(A, 0)
        CNV = max(CNV, 0)
        SNV = max(SNV, 0)
        S = max(S, 0)
        S = min(S, S_init)
        t+=τ
        states.append(np.array([A, CNV, SNV, S]))
        times.append(t)
    
    data = np.stack(states, axis=1)
    times=np.array(times)
    gens=times/5.8 #convert hours to generations
    #get the indices closest to the generations observed
    ngens = len(generation)
    indices = np.empty(ngens, dtype=int)
    for i in range(ngens):
        ab = np.abs(gens-generation[i])
        indices[i] = ab.argmin()
    # convert pop densities for each of three states to cnv proportion
    p_cnv = data[1,:] / data[0:3,:].sum(axis=0)
    
    return np.transpose(p_cnv)[indices]

def CNVsimulator_simpleChemo(reps, s_snv, m_snv, generation, parameters, seed=None):
    """ Simulates CNV and SNV evolution in a steady state chemostat for some generations
    Begins counting generations after 48 hours, during which time the chemostat reaches steady state
    Returns proportion of the population with a CNV for generations observed in Lauer et al. 2018 as 1d np.array of length len(generation)
    reps: number of replicate simulations to perform
    generation : np.array, 1d 
        with generations to output
    A_inoculation :  cell density at inoculation with ancestral type
    S_init : initial substrate concentration in chemostat
    k : substrate concentration at half-maximal growth rate (here, the same for all cell types)
    D : dilution rate
    μA : ancestral maximum growth rate
    m_snv : ancestral -> SNV mutation rate
    s_snv : SNV selection coefficient
    updates : change in reactants due to reactions
    S0 : incoming substrate concentration - S0 in diff eq
    y : number of cells produced per mole of the limiting nutrient (here, the same for all cell types)
    τ : amount to advance time
    m_cnv : ancestral -> CNV mutation rate
    s_cnv : CNV selection coefficient
    seed : int
    parameters : instance of parameters
        has attribute s, float, log 10 selection coefficient
        has attribute m, float, log 10 cnv mutation rate
        for use with pyABC
        OR
        np.array, 1d of length dim_param
        Parameter vector with the log 10 selection coefficient and log 10 cnv mutation rate, for use with SNPE or to build observed data
    """
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()
    s_cnv, m_cnv = get_params(parameters) 
    evo_reps = []
    for i in range(reps):
        out=simpleChemo(generation, A_inoculation, S_init, k, D, μA, m_snv, s_snv, S0, y, τ, s_cnv, m_cnv, seed=seed)
        evo_reps.append(out)
    return np.array(evo_reps)

def CNVsimulator_multiChemo(reps, s_snv, m_snv, generation, parameters, return_s = False, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for some generations
    Returns proportion of the population with a CNV for generation for n_pops populations as 2d np.array of generation x n_pop
    
    Parameters
    -------------------
    reps: int
        number of replicate simulations to perform
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
     parameters : instance of parameters
        has attribute gamma_shape, float, parameterizes gamma distribution that CNV selection coefficient is drawn from
        has attribute log10 gamma_scale, float, parameterizes gamma distribution that CNV selection coefficient is drawn from
        has attribute m, float, log 10 cnv mutation rate
        for use with pyABC
        OR
        np.array, 1d of length dim_param
        Parameter vector with the gamma_shape, log10 gamma_scale, and log 10 cnv mutation rate, for use with SNPE or to build observed data
    return_s : Boolean
    """
    all_obs = []
    shape, scale, m_cnv = get_params_multi(parameters)
    s_cnv = np.random.gamma(shape=shape, scale=scale, size=reps)
    for i in range(reps):
        obs=simpleChemo(generation, A_inoculation, S_init, k, D, μA, m_snv, s_snv, S0, y, τ, s_cnv=s_cnv[i], m_cnv=m_cnv, seed=seed)  
        all_obs.append(obs)
    if return_s == False:
        return np.vstack(all_obs)
    if return_s == True:
        return np.vstack(all_obs),s_cnv