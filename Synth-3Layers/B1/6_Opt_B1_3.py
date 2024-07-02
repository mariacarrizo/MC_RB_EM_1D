# Script to perform gradient based inversion on 3-layered 1D models

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Load forward modelling classes for 3-layered 1D models
from EM1D import EMf_3Lay_Opt_HVP_1D, EMf_3Lay_Opt_HVP_Q_1D, EMf_3Lay_Opt_HVP_IP_1D

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model_B1_1 = np.load('models/model_synth_B1_1.npy')
model_B1_2 = np.load('models/model_synth_B1_2.npy')
model_B1_3 = np.load('models/model_synth_B1_3.npy')
model_B1_4 = np.load('models/model_synth_B1_4.npy')

# Data array for all the 1D stitched models
data_B1_1 = np.load('data/data_synth_B1_1.npy')
data_B1_2 = np.load('data/data_synth_B1_2.npy')
data_B1_3 = np.load('data/data_synth_B1_3.npy')
data_B1_4 = np.load('data/data_synth_B1_4.npy')

npos = len(data_B1_1) # number of 1D models

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

m0 = [3, 3, 200/1000, 200/1000, 200/1000]
lam = 0

transThk = pg.trans.TransLogLU(np.min(thick), np.max(thick))
transSig = pg.trans.TransLogLU(np.min(conds), np.max(conds))
transData = pg.trans.TransLog()

# Optimization Q + IP

print('Estimating model B1-3 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)
EMf.transData = transData

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_3[0]) * error
model_Opt_B1_3 = np.zeros_like(model_B1_3)
#model_hist = []

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    print('pos:', pos)
    dataE = data_B1_3[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_Opt_B1_3[pos] = model_Opt_pos
    if pos == 10:
        model_hist = invEM.modelHistory
print('End')

