# cnv_sims_inference
CNV simulation and inference of formation rate and selection coefficient

Inference:
* Models:
  * Wright-Fisher and Chemostat: cnv_simulation.py
  * Determining the effective population size in the chemostat: Pop_sampling_variance_sims.ipynb
* Observations:
  * Single synthetic observations: generate_pseudo_obs.py
  * Sets of multiple synthetic observations: Generate_synthetic_obs_multi.ipynb
  * Interpolation of barcoded population data (so that it has the same timepoints as gln01-gln09): Interpolating_bc.ipynb
* Scripts used for inference:
  * SMC-ABC (using pyABC, adaptive Euclidean distance): infer_pyABC.py
  * NPE, single observations (using sbi): infer_sbi.py
  * NPE, sets of multiple observations (using sbi): infer_sbi_mult.py
  * NPE, on empirical data from Lauer et al 2018 (using sbi): infer_sbi_Lauer.py
  
Barcode DFE:  
Note, population bc04 is bc01 in the paper.  
* Extract barcodes from fastqs and cluster them (using bartender): Barcode_sequencing.Rmd
* Barcode DFE inference overview as well as checks for mean fitness convergence, etc., and supplementary figure 13: 2021-09-16_analysis_Grace.ipynb
* Barcode DFE inference detailed: fitmut_v2_a_20210916.py

Fitness assays:
* Fitting models and extracting selection coefficients from competitions between CNV containing clones and the ancestral strain: fitness_assays.R

Figures:
* Figure 1A: Fig1A.R
* Figure 1D inset: Fig1Dinset.ipynb
* Figures 2-6, and associated supplemental material (html associated with each Rmd): Figure2andSup.Rmd, Figure3andSup.Rmd, Figure4andSup.Rmd, Figure5andSup.Rmd, Figure6andSup.Rmd




