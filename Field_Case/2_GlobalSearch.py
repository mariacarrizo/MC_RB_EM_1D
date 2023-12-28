# Import libraries

import numpy as np
import time
import pandas as pd

import sys
sys.path.insert(1, '../src')

# Load function that performs global search in lookup table
from EM1D import GlobalSearch_2Lay

# Load lookup table and sampling 
LUT = np.load('data/LUTable_2Lay_field.npy')
conds = np.load('data/conds.npy')
thicks =  np.load('data/thicks.npy')

# Load field data 
Dataframe = pd.DataFrame(np.load('data/Field_data.npy'),
                        columns = ['X','Y','Position','Z','H2Q','H4Q','H8Q',
                                   'V2Q','V4Q','V8Q','P2Q','P4Q','P8Q',
                                   'H4IP','H8IP','V4IP','V8IP'])

data = np.array(pd.concat([Dataframe.loc[:,'H2Q':'V8Q'], Dataframe.loc[:,'H4IP':]], axis=1))

# number of 1D positions
npos = len(data)

# Estimate with both Quadrature and In Phase
model_est = [] # Empty array for estimated model

print('Starting global search ...')

starttime = time.time()
for p in range(npos):
    model_est.append(GlobalSearch_2Lay(LUT, data[p], conds, thicks, nsl=len(conds)))
endtime = time.time() - starttime

print('Global search Q+IP excution for ', npos, ' positions: ', f"{(endtime/60):.3}", 'minutes')

# Save estimated models
np.save('results/model_2Lay_GS_field', model_est)
