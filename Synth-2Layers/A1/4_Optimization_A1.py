import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

from EM1D import EMf_2Lay_Opt_HVP, EMf_2Lay_Opt_HVP_Q, EMf_2Lay_Opt_HVP_IP

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model = np.load('data/model_synth_2Lay_A1.npy')

# Data array for all the 1D stitched models
data = np.load('data/data_synth_2Lay_A1.npy')
npos = len(data) # number of positions

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Optimization Q + IP

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # introduce here the error you want to test
relativeError = np.ones_like(data[0]) * error
model_est = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos].copy()
    #dataE *= np.random.randn(len(dataE)) * relativeError + 1.0
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est[pos] = model_est_pos
    
# Optimization Q 

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP_Q(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # introduce here the error you want to test
relativeError = np.ones_like(data[0, :9]) * error
model_est_Q = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, :9].copy()
   # dataE *= np.random.randn(len(dataE)) * relativeError + 1.0
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est_Q[pos] = model_est_pos
    
# Optimization IP 

# Initialize the forward modelling class
EMf = EMf_2Lay_Opt_HVP_IP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # introduce here the error you want to test
relativeError = np.ones_like(data[0, 9:]) * error
model_est_IP = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos, 9:].copy()
    # dataE *= np.random.randn(len(dataE)) * relativeError + 1.0
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est_IP[pos] = model_est_pos

# Save estimates
np.save('results/model_2Lay_A1_Opt', model_est)
np.save('results/model_2Lay_A1_Opt_Q', model_est_Q)
np.save('results/model_2Lay_A1_Opt_IP', model_est_IP)