import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

from EM1D import EMf_3Lay_Opt_HVP

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

# Import true models and data 
model = np.load('data/model_synth_3Lay_B1.npy')

# Data array for all the 1D stitched models
data = np.load('data/data_synth_3Lay_B1.npy')
npos = len(data) # number of positions

# Load data with added noise
data_noise_2 = np.load('data/data_B1_n2.npy')
data_noise_5 = np.load('data/data_B1_n5.npy')
data_noise_10 = np.load('data/data_B1_n10.npy')

# Optimization Q + IP noise 2.5 %
# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array
error = 1e-3 # introduce here the error you want to test
relativeError = np.ones_like(data[0]) * error
model_n2 = np.zeros_like(model)

# Setting a lower boundary of conductivities 10 mS/m
transModel = pg.trans.TransLogLU(0.01,7) 
invEM.modelTrans = transModel

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    model_est_pos = invEM.run(data_noise_2[pos], relativeError, verbose=False)
    model_n2[pos] = model_est_pos
#    if (model_n2[pos, (model_n2[pos,0] >1)]).any():
#        model_n2[pos,0] = 1
#    if (model_n2[pos, (model_n2[pos,1] >1)]).any():
#        model_n2[pos,1] = 1
#    if (model_n2[pos, (model_n2[pos,2] >1)]).any():
#        model_n2[pos,2] = 1
#    if (model_n2[pos, (model_n2[pos,3] >10)]).any():
#        model_n2[pos,3] = 10
#    if (model_n2[pos, (model_n2[pos,3] >10)]).any():
#        model_n2[pos,4] = 10
        
# Optimization Q + IP noise 5 %
# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Setting a lower boundary of conductivities 10 mS/m
transModel = pg.trans.TransLogLU(0.01,7) 
invEM.modelTrans = transModel

model_n5 = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    model_est_pos = invEM.run(data_noise_5[pos], relativeError, verbose=False)
    model_n5[pos] = model_est_pos
#    if (model_n5[pos, (model_n5[pos,0] >1)]).any():
#        model_n5[pos,0] = 1
#    if (model_n5[pos, (model_n5[pos,1] >1)]).any():
#        model_n5[pos,1] = 1
#    if (model_n5[pos, (model_n5[pos,2] >1)]).any():
#        model_n5[pos,2] = 1
#    if (model_n5[pos, (model_n5[pos,3] >10)]).any():
#        model_n5[pos,3] = 10
#    if (model_n5[pos, (model_n5[pos,3] >10)]).any():
#        model_n5[pos,4] = 10
        
# Optimization Q + IP noise 10 %
# Initialize the forward modelling class
EMf = EMf_3Lay_Opt_HVP(lambd, height, offsets, freq, filt)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Setting a lower boundary of conductivities 10 mS/m
transModel = pg.trans.TransLogLU(0.01,7) 
invEM.modelTrans = transModel

model_n10 = np.zeros_like(model)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    model_est_pos = invEM.run(data_noise_10[pos], relativeError, verbose=False)
    model_n10[pos] = model_est_pos
#    if (model_n10[pos, (model_n10[pos,0] >1)]).any():
#        model_n10[pos,0] = 1
#    if (model_n10[pos, (model_n10[pos,1] >1)]).any():
#        model_n10[pos,1] = 1
#    if (model_n10[pos, (model_n10[pos,2] >1)]).any():
#        model_n10[pos,2] = 1
#    if (model_n10[pos, (model_n10[pos,3] >10)]).any():
#        model_n10[pos,3] = 10
#    if (model_n10[pos, (model_n10[pos,3] >10)]).any():
#        model_n10[pos,4] = 10
        
# Save estimates
np.save('results/model_3Lay_B1_Opt_n2', model_n2)
np.save('results/model_3Lay_B1_Opt_n5', model_n5)
np.save('results/model_3Lay_B1_Opt_n10', model_n10)