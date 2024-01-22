# Script to perform gradient based inversion on 3-layered 1D models

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Load forward modelling classes for 3-layered 1D models
from EM1D import EMf_3Lay_Opt_HVP, EMf_3Lay_Opt_HVP_Q, EMf_3Lay_Opt_HVP_IP

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model = np.load('data/model_synth_3Lay_B1.npy')

# Data array for all the 1D stitched models
data = np.load('data/data_synth_3Lay_B1.npy')
npos = len(data) # number of 1D models

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

#%%
# Optimization Q + IP

# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0]) * error
model_opt = np.zeros_like(model)

# Setting a lower boundary of conductivities 1 mS/m
transModel = pg.trans.TransLogLU(0.001,7) 
invEM.modelTrans = transModel

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos].copy()
    model_opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_opt[pos] = model_opt_pos

#%%
# Optimization Q 

# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP_Q(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0, :9]) * error
model_opt_Q = np.zeros_like(model)

# Setting a lower boundary of conductivities 1 mS/m
transModel = pg.trans.TransLogLU(0.001,7) 
invEM.modelTrans = transModel

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, :9].copy()
    model_opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_opt_Q[pos] = model_opt_pos
    
#%%    
# Optimization IP 

# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP_IP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0, 9:]) * error
model_opt_IP = np.zeros_like(model)

# Setting a lower boundary of conductivities 1 mS/m
transModel = pg.trans.TransLogLU(0.001,7) 
invEM.modelTrans = transModel

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, 9:].copy()
    model_opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_opt_IP[pos] = model_opt_pos

#%%        
# Save estimated models
np.save('results/model_3Lay_B1_Opt', model_opt)
np.save('results/model_3Lay_B1_Opt_Q', model_opt_Q)
np.save('results/model_3Lay_B1_Opt_IP', model_opt_IP)