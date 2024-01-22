### Code that creates searches in Lookup table for the indices of best data fit 
### (min data misfit) for 3-layered 1D models using Quadrature

## Import libraries
import numpy as np
import time
from joblib import Parallel, delayed
import sys
path = '../../src'
sys.path.insert(0, path)

# Import global search function
from EM1D import GlobalSearch_3Lay

# Load survey information
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()

offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Normalize by offset
norm = np.hstack((offsets, offsets, offsets, offsets, offsets, offsets))

# number of cores used to perform the global search
n_workers=8

# Load conductivities and thicknesses sampled
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')
nsl = len(conds)

# Load lookup table
LUT = np.load('../data/LUTable_3Lay.npy')

## Load true synthetic model and data
data = np.load('data/data_synth_3Lay_B2.npy')
npos = len(data)

LUT_norm = LUT[:]*norm
data_norm = data[:]*norm

# Start inversion
print('Started searching error vector using Lookup table ...')
startTime = time.time()

model = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT_norm[:,:9], data_norm[pos,:9],
    conds, thicks, norm[:9]) for pos in range(npos))

executionTime = (time.time() - startTime)/60
print('Execution time in seconds: ', f"{executionTime:.3}", ' minutes')

# Save the estimated model
np.save('results/model_3Lay_B2_GS_Q', model)





