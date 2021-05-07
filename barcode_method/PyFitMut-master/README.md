[![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Contact Info](https://img.shields.io/badge/Contact%20Info-fangfeili0525@gmail.com-orange.svg)]()
[![Update Info](https://img.shields.io/badge/Update%20Info-Jan12,%202021-orange.svg)]()


## PyFitMut

### 1. What is PyFitMut?

PyFitMut is a Python-based tool that can identify spontaneous adaptive mutations for initially isogenic evolving population, as well as estimate the fitness effect and establishment time of those adaptive mutations. The detailed theory and algorithm of PyFitMut is introduced in reference: [S. F. Levy, et al. Quantitative Evolutionary Dynamics Using High-resolution Lineage Tracking. Nature, 519: 181-186 (2015)](https://www.nature.com/articles/nature14279). If you use this software, please reference: [?](?). PyFitMut is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 

It currently has one main function:
* `pyfitmut_v10.py` calculates the fitness and establishment time of all adaptive mutations from read-count time-series data.
    
A walk-through is included as the jupyter notebook [here](https://github.com/FangfeiLi05/PyFitMut/blob/master/PyFitMut_walk_through.ipynb).



### 2. How to install PyFitMut?
* Python 3 is required. This version has been tested on a MacBook Pro (3.1 GHz Intel Core i5), with Python 3.7.4.
* Clone this repository by running `git clone https://github.com/FangfeiLi05/PyFitMut.git` in terminal.
* `cd` to the root directory of the project (the folder containing `README.md`).
* Install dependencies by running `pip install -r requirements.txt` in terminal.



### 3. How to use PyFitMut?

`pyfitmut_v10.py` identifies adaptive mutations and estimates their fitness effect as well as establishment time from read-count time-series data.


##### OPTIONS
* `--input` or `-i`: a .csv file, with each column being the read number per barcode at each sequenced time-point
* `--t_seq` or `-t`: a .csv file of 2 columns: 1st column is sequenced time-points evaluated in number of generations, 2nd column is total number of cells of the population (effective) for each sequenced time-point.
* `--mutation_rate` or `-u`: total beneficial mutation rate (`default: 1e-5`)
* `--c` or `-c`: a noise parameter that characterizes the cell growth and cell transfer (`default: 0.5`)
* `--output_filename` or `-o`: prefix of output .csv files (`default: output`)


##### OUTPUTS
* `output_filename_FitSeq_Result.csv`: a .csv file, with
  + 1st column of .csv: estimated fitness of each lineage (0 if there is no mutation), [x1, x2, ...]
  + 2nd column of .csv: estimated estabblishment time of each lineage (0 if there is no mutation), [tau1, tau2, ...]
  + 3rd column of .csv: log likelihood value of each lineage (0 if there is no mutation), [f1, f2, ...]
  + 4rd column of .csv: log likelihood value (adaptive) of each lineage (0 if there is no mutation), [f1, f2, ...]
  + 5rd column of .csv: log likelihood value (neutral) of each lineage (0 if there is no mutation), [f1, f2, ...]
  + 6rd column of .csv: estimated mean fitness per sequenced time-point, [x_mean(0), x_mean(t1), ...]
  + 7rd column of .csv: estimated kappa value per sequenced time-point, [kappa(0), kappa(t1), ...]

##### For Help
```
python pyfitmut.py --help
```  

##### Examples
```  
python pyfitmut_v10.py -i test_EvoSimulation_Read_Number.csv -t input_sequenced_time_points.csv -u 2e-5 -o test
``` 




