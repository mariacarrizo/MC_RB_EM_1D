import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

from EM1D import EMf_3Lay_Opt_HVP, EMf_3Lay_Opt_HVP_Q, EMf_3Lay_Opt_HVP_IP

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model = np.load('data/model_synth_3Lay_B2.npy')

# Data array for all the 1D stitched models
data = np.load('data/data_synth_3Lay_B2.npy')
npos = len(data) # number of positions

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Optimization Q + IP

# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # introduce here the error you want to test
relativeError = np.ones_like(data[0]) * error
model_est = np.zeros_like(model)

# Setting a lower boundary of conductivities 10 mS/m
transModel = pg.trans.TransLogLU(0.01,7) 
invEM.modelTrans = transModel

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    dataE = data[pos].copy()
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est[pos] = model_est_pos
#    if (model_est[pos, (model_est[pos,0] >1)]).any():
#        model_est[pos,0] = 1
#    if (model_est[pos, (model_est[pos,1] >1)]).any():
#        model_est[pos,1] = 1
#    if (model_est[pos, (model_est[pos,2] >1)]).any():
#        model_est[pos,2] = 1
#    if (model_est[pos, (model_est[pos,3] >10)]).any():
#        model_est[pos,3] = 10
#    if (model_est[pos, (model_est[pos,3] >10)]).any():
#        model_est[pos,4] = 10
        
# Optimization Q 

# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP_Q(lambd, height, offsets, freq, filt)

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
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est_Q[pos] = model_est_pos
#    if (model_est_Q[pos, (model_est_Q[pos,0] >1)]).any():
#        model_est_Q[pos,0] = 1
#    if (model_est_Q[pos, (model_est_Q[pos,1] >1)]).any():
#        model_est_Q[pos,1] = 1
#    if (model_est_Q[pos, (model_est_Q[pos,2] >1)]).any():
#        model_est_Q[pos,2] = 1
#    if (model_est_Q[pos, (model_est_Q[pos,3] >10)]).any():
#        model_est_Q[pos,3] = 10
#    if (model_est_Q[pos, (model_est_Q[pos,3] >10)]).any():
#        model_est_Q[pos,4] = 10
        
# Optimization IP 

# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP_IP(lambd, height, offsets, freq, filt)

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
    model_est_pos = invEM.run(dataE, relativeError, verbose=False)
    model_est_IP[pos] = model_est_pos
#    if (model_est_IP[pos, (model_est_IP[pos,0] >1)]).any():
#        model_est_IP[pos,0] = 1
#    if (model_est_IP[pos, (model_est_IP[pos,1] >1)]).any():
#        model_est_IP[pos,1] = 1
#    if (model_est_IP[pos, (model_est_IP[pos,2] >1)]).any():
#        model_est_IP[pos,2] = 1
#    if (model_est_IP[pos, (model_est_IP[pos,3] >10)]).any():
#        model_est_IP[pos,3] = 10
#    if (model_est_IP[pos, (model_est_IP[pos,3] >10)]).any():
#        model_est_IP[pos,4] = 10
        
# Save estimates
np.save('results/model_3Lay_B2_Opt', model_est)
np.save('results/model_3Lay_B2_Opt_Q', model_est_Q)
np.save('results/model_3Lay_B2_Opt_IP', model_est_IP)