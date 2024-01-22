# Script that performs gradient based inversion for Field case

# Import libraries
import pygimli as pg
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../src')

# Load forward modelling class for 2-layered models in field case
from EM1D import EMf_2Lay_Opt_HV_field

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('data/conds.npy')
thick = np.load('data/thicks.npy')

# Import field data
Dataframe = pd.DataFrame(np.load('data/Field_data.npy'),
                        columns = ['X','Y','Position','Z','H2Q','H4Q','H8Q',
                                   'V2Q','V4Q','V8Q','P2Q','P4Q','P8Q',
                                   'H4IP','H8IP','V4IP','V8IP'])

# Obtain H and V quadrature and in-phase measurements
# For in-phase we only use measurements for offsets > 4 m
data = np.array(pd.concat([Dataframe.loc[:,'H2Q':'V8Q'], Dataframe.loc[:,'H4IP':]], axis=1))
npos = len(data) # number of 1D models
nlay = 2 # number of layers

# Load survey parameters
survey = np.load('data/survey_field.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Optimization Q + IP

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HV_field(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Setting a lower boundary of conductivities 10 mS/m

transModel = pg.trans.TransLogLU(0.01,6) 
invEM.modelTrans = transModel

# Relative error array
error = 1e-3 # relative error 1e-3 
relativeError = np.ones_like(data[0]) * error
model_est = np.zeros((npos, nlay+1))

# Start inversion
# Perform inversion for each 1D model per position 
for pos in range(npos):
    dataE = data[pos].copy()
    model_est[pos] = invEM.run(dataE, relativeError, verbose=False)

# Save estimated model
np.save('results/model_2Lay_Opt_field', model_est)