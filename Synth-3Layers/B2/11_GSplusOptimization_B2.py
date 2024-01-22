# Script that performs the combined algorithm of global search plus gradient 
# based inversion for 3-layered 1D models case B.2 

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# import forward modelling class for 3-layered 1D models for GS + Opt algorithm
from EM1D import EMf_3Lay_GSplusOpt_HVP

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Data array for all the 1D stitched models
data = np.load('data/data_synth_3Lay_B2.npy')
npos = len(data) # number of 1D models

# Import model from Global search
model_GS = np.load('results/model_3Lay_B2_GS.npy')

# Optimization Q + IP

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0]) * error
model_est = np.zeros_like(model_GS)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_GSplusOpt_HVP(lambd, height, offsets, freq, filt, m0)

    # Create inversion
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf)
    
    # Setting a lower boundary of conductivities 1 mS/m
    #transModel = pg.trans.TransLogLU(0.001,7) 
    #invEM.modelTrans = transModel

    dataE = data[pos].copy()
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est[pos] = model_est_pos

# Store estimated model        
np.save('results/model_3Lay_GSplusOpt_B2', model_est)