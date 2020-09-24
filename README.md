# cnv_sims_inference
CNV simulation and inference of mutation rate and fitness effect

Notebooks with code and examples of output:
* Models:
  * Wright-Fisher: WF_model.ipynb
  * Chemostat: chemo_model.ipynb
* Inference:
  * ABC (using pyABC, adaptive Euclidean distance): cnv_pyABC.ipynb
  * APT (using delfi): cnv_delfi.ipynb
  
Scripts for use on the HPC:
* cnv_simulation.py
* infer_pyABC.py
* infer_delfi.py
* BATCH_cnv_sim_pyABC.sh
* BATCH_cnv_sim_delfi.sh

Other:
* performance_of_algorithms.R
* cnv_WAIC.ipynb

