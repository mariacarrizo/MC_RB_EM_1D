# Script that performs the combined algorithm of global search + 
# gradient based inversion for 3-layered 1D models

# Import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling class for 3-layered 1D models for GS + Opt algorithm
from EM1D import EMf_3Lay_Opt_HVP_1D

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
data_B1_1 = np.load('data/data_synth_B2_1.npy')
data_B1_2 = np.load('data/data_synth_B2_2.npy')
data_B1_3 = np.load('data/data_synth_B2_3.npy')
data_B1_4 = np.load('data/data_synth_B2_4.npy')

npos = len(data_B1_1) # number of 1D models

# Import model from Global search
model_GS_B1_1 = np.load('results/model_GS_B2_1.npy')
model_GS_B1_2 = np.load('results/model_GS_B2_2.npy')
model_GS_B1_3 = np.load('results/model_GS_B2_3.npy')
model_GS_B1_4 = np.load('results/model_GS_B2_4.npy')

transThk = pg.trans.TransLogLU(np.min(thick), np.max(thick))
transSig = pg.trans.TransLogLU(np.min(conds), np.max(conds))
lam=0

#%%
# Optimization Q + IP

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_1[0]) * error
model_est_B1_1 = np.zeros_like(model_GS_B1_1)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B1_1[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    #EMf.transData = transData
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B1_1[pos].copy()
    model_est_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_est_B1_1[pos] = model_est_pos

# Save estimated model    
np.save('results/model_GSGN_B2_1', model_est_B1_1)

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_2[0]) * error
model_est_B1_2 = np.zeros_like(model_GS_B1_2)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B1_2[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    #EMf.transData = transData
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B1_2[pos].copy()
    model_est_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_est_B1_2[pos] = model_est_pos

# Save estimated model    
np.save('results/model_GSGN_B2_2', model_est_B1_2)

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_3[0]) * error
model_est_B1_3 = np.zeros_like(model_GS_B1_3)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B1_3[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    #EMf.transData = transData
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B1_3[pos].copy()
    model_est_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_est_B1_3[pos] = model_est_pos

# Save estimated model    
np.save('results/model_GSGN_B2_3', model_est_B1_3)

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_4[0]) * error
model_est_B1_4 = np.zeros_like(model_GS_B1_4)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B1_4[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    #EMf.transData = transData
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B1_4[pos].copy()
    model_est_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_est_B1_4[pos] = model_est_pos

# Save estimated model    
np.save('results/model_GSGN_B2_4', model_est_B1_4)
