import pygimli as pg
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../src')

from EM1D import EMf_2Lay_GSplusOpt_HV_field

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('data/conds.npy')
thick = np.load('data/thicks.npy')

# Import data
Dataframe = pd.DataFrame(np.load('data/Field_data.npy'),
                        columns = ['X','Y','Position','Z','H2Q','H4Q','H8Q',
                                   'V2Q','V4Q','V8Q','P2Q','P4Q','P8Q',
                                   'H4IP','H8IP','V4IP','V8IP'])

data = np.array(pd.concat([Dataframe.loc[:,'H2Q':'V8Q'], Dataframe.loc[:,'H4IP':]], axis=1))
npos = len(data) # number of positions
nlay = 2

# Load survey parameters
survey = np.load('data/survey_field.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Import model from Global search
model_GS = np.load('results/model_2Lay_GS_field.npy')

# Optimization Q + IP

# Relative error array
error = 1e-3 # introduce here the error you want to test
relativeError = np.ones_like(data[0]) * error
model_est = np.zeros((npos, nlay+1))

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_2Lay_GSplusOpt_HV_field(lambd, height, offsets, freq, filt, m0)

    # Create inversion
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf)
    
    # Setting a lower boundary of conductivities 0.9 mS/m
    transModel = pg.trans.TransLogLU(0.01,6) 
    invEM.modelTrans = transModel

    dataE = data[pos].copy()
    model_est[pos] = invEM.run(dataE, relativeError, verbose=False)
    
np.save('results/model_2Lay_GSplusOpt_field', model_est)