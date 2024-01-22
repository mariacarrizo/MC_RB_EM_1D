# Script to perform gradient based inversion for case A.2

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling classes for 2-layered models
from EM1D import EMf_2Lay_Opt_HVP, EMf_2Lay_Opt_HVP_Q, EMf_2Lay_Opt_HVP_IP

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model = np.load('data/model_synth_2Lay_A2.npy')
data = np.load('data/data_synth_2Lay_A2.npy')

# number of 1D models
npos = len(data) 

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

#%%
# Optimization Q + IP

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Define inversion framework
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0]) * error

# Create array to store estimation
model_Opt = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position 
for pos in range(npos):
    dataE = data[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_Opt[pos] = model_Opt_pos

#%%
# Optimization Q 

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP_Q(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 
relativeError = np.ones_like(data[0, :9]) * error

# Create array to store estimation
model_Opt_Q = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_Opt_Q[pos] = model_Opt_pos

#%%
# Optimization IP 

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP_IP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3
relativeError = np.ones_like(data[0, 9:]) * error

# Create array to store estimation
model_Opt_IP = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_Opt_IP[pos] = model_Opt_pos

#%%
# Save estimated models
np.save('results/model_2Lay_A2_Opt', model_Opt)
np.save('results/model_2Lay_A2_Opt_Q', model_Opt_Q)
np.save('results/model_2Lay_A2_Opt_IP', model_Opt_IP)