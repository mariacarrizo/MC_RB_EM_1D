# Script to perform gradient based inversion for case A.1

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling classes for 2-layered models
from EM1D import EMf_2Lay_Opt_HVP_1D, EMf_2Lay_Opt_HVP_Q, EMf_2Lay_Opt_HVP_IP

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model_A1_1 = np.load('data/model_synth_2Lay_A1_1.npy')
model_A1_2 = np.load('data/model_synth_2Lay_A1_2.npy')
model_A1_3 = np.load('data/model_synth_2Lay_A1_3.npy')
model_A1_4 = np.load('data/model_synth_2Lay_A1_4.npy')

data_A1_1 = np.load('data/data_synth_2Lay_A1_1.npy')
data_A1_2 = np.load('data/data_synth_2Lay_A1_2.npy')
data_A1_3 = np.load('data/data_synth_2Lay_A1_3.npy')
data_A1_4 = np.load('data/data_synth_2Lay_A1_4.npy')

# Number of 1D models
npos = len(data_A1_1) # number of positions

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

#%%
# Optimization Q + IP

print('Estimating model A1-1 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Define transformation
transModel = pg.trans.TransLogLU(0.01,7) 

# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator
invEM.modelTrans = transModel

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0]) * error
model_Opt_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_Opt_A1_1[pos] = model_Opt_pos
    
print('Estimating model A1-2 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Define transformation
transModel = pg.trans.TransLogLU(0.01,7) 

# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator
invEM.modelTrans = transModel

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data[0]) * error
model_Opt_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_Opt_A1_2[pos] = model_Opt_pos

#%%
# Optimization Q 

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP_Q(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # relative error 
relativeError = np.ones_like(data[0, :9]) * error
model_Opt_Q = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model 
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
error = 1e-3 # relative error
relativeError = np.ones_like(data[0, 9:]) * error
model_Opt_IP = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, verbose=False)
    model_Opt_IP[pos] = model_Opt_pos

#%%
# Save estimates
np.save('results/model_2Lay_A1_Opt', model_Opt)
np.save('results/model_2Lay_A1_Opt_Q', model_Opt_Q)
np.save('results/model_2Lay_A1_Opt_IP', model_Opt_IP)