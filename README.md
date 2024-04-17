# CNV simulation and inference of formation rate and selection coefficient

This is the repository for the paper:
> Avecilla, Grace, Julie N Chuong, Fangfei Li, Gavin Sherlock, David Gresham, and Yoav Ram. [**Neural Networks Enable Efficient and Accurate Simulation-Based Inference of Evolutionary Parameters from Adaptation Dynamics.**](https://doi.org/10.1371/journal.pbio.3001633) _PLOS Biology_ 20, no. 5 (May 27, 2022): e3001633. doi:10.1371/journal.pbio.3001633.

## Data
Data to generate figures can be found at [OSF](https://osf.io/e9d5x/).

## Inference:
* Models:
  * Wright-Fisher and Chemostat: `cnv_simulation.py`
  * Determining the effective population size in the chemostat: `Pop_sampling_variance_sims.ipynb`
  * Time it takes to run a simulation using each model: `Simulation_time.ipynb`
* Observations:
  * Single synthetic observations: `generate_pseudo_obs.py`
  * Sets of multiple synthetic observations: `Generate_synthetic_obs_multi.ipynb`
  * Interpolation of barcoded population data (so that it has the same timepoints as gln01-gln09): `Interpolating_bc.ipynb`
* Scripts used for inference:
  * Generating presimulated data used for rejection ABC and NPE: `generate_presimulated_data.py`
  * Rejection ABC: `infer_rejectionABC.py`
  * SMC-ABC (using `pyABC`, adaptive Euclidean distance): `infer_pyABC.py`
  * NPE, single observations (using `sbi`): `infer_sbi.py`
  * NPE, sets of multiple observations (using `sbi`): `infer_sbi_mult.py`
  * NPE, on empirical data from Lauer et al 2018 (using `sbi`): `infer_sbi_Lauer.py`
  
Barcode DFE:  
Note, population _bc04_ is _bc0_1 in the paper.  
* Extract barcodes from fastqs and cluster them (using bartender): get_bc.sh
* Combines barcode counts from different timepoints: combine_bc.sh
* Barcode DFE inference overview as well as checks for mean fitness convergence, etc., and supplementary figure 13: 2021-09-16_analysis_Grace.ipynb
* Barcode DFE inference detailed: fitmut_v2_a_20210916.py

Fitness assays:
* Fitting models and extracting selection coefficients from competitions between CNV containing clones and the ancestral strain: `fitness_assays.R`

Figures:
* Figure 1A: `Fig1A.R`
* Supplementary Figure 1: `Interpolating_bc.ipynb`
* Figure 1D inset: `Fig1Dinset.ipynb`
* Figures 3-7, and associated supplemental material (html associated with each Rmd): `Figure3andSup.Rmd`, `Figure4andSup.Rmd`, `Figure5andSup.Rmd`, `Figure6andSup.Rmd`, `Figure7andSup.Rmd`




