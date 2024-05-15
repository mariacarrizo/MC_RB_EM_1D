### Code that creates searches in Lookup table for the indices of best data fit 
### (min data misfit) for 3-layered 1D models

## Import libraries
import numpy as np
import time
from joblib import Parallel, delayed
import sys
path = '../../src'
sys.path.insert(0, path)

# Import global search function for 3-layered models
from EM1D import GlobalSearch_3Lay

# Load survey information
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()

offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# number of cores used to perform the global search
n_workers=48

# Load conductivities and layer thicknesses sampled
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')
nsl = len(conds) # number of samples

# Load lookup table
LUT = np.load('../data/LUTable_3Lay.npy')

## Load true synthetic model and data
data_B2_1 = np.load('data/data_synth_B2_1.npy')
data_B2_2 = np.load('data/data_synth_B2_2.npy')
data_B2_3 = np.load('data/data_synth_B2_3.npy')
data_B2_4 = np.load('data/data_synth_B2_4.npy')

npos = len(data_B2_1)

# Start global search inversion
print('Started global search for model B2-1 ...')
startTime = time.time()

model_GS_B2_1 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT, 
                         data_B2_1[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

print('Started global search for model B2-2 ...')
startTime = time.time()

model_GS_B2_2 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT, 
                         data_B2_2[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

print('Started global search for model B2-3 ...')
startTime = time.time()

model_GS_B2_3 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT, 
                         data_B2_3[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

print('Started global search for model B2-4 ...')
startTime = time.time()

model_GS_B2_4 = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT, 
                         data_B2_4[pos], conds, thicks) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time: ', f"{executionTime:.3}", ' minutes')

# Save estimated model
np.save('results/model_GS_B2_1', model_GS_B2_1)
np.save('results/model_GS_B2_2', model_GS_B2_2)
np.save('results/model_GS_B2_3', model_GS_B2_3)
np.save('results/model_GS_B2_4', model_GS_B2_4)






