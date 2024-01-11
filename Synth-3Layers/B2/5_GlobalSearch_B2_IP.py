### Code that creates searches in Lookup table for the indices of best data fit (min error)
### 1D 3 layered model

## Import libraries

import numpy as np
import time
from joblib import Parallel, delayed
from itertools import product

import sys
path = '../../src'
sys.path.insert(0, path)

# Import global search function
from EM1D import GlobalSearch_3Lay

n_workers=4

# Load conductivities and thicknesses sampled
conds = np.load('../data/conds.npy')
thicks = np.load('../data/thicks.npy')
nsl = len(conds)

# Load lookup table
LUT = np.load('../data/LUTable_3Lay.npy')

## Load true synthetic model and data
data = np.load('data/data_synth_3Lay_B2.npy')
npos = len(data)

# Start inversion
print('Started searching error vector using Lookup table ...')
startTime = time.time()

model = Parallel(n_jobs=n_workers,verbose=0)(delayed(GlobalSearch_3Lay)(LUT[:,9:], data[pos,9:],
                                                                 conds, thicks) for pos in range(npos))

executionTime = time.time() - startTime
print('Execution time in seconds: ' + str(executionTime))

np.save('results/model_3Lay_B2_GS_IP', model)





