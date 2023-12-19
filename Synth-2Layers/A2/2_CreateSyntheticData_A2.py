# File to create synthetic model

# Import libraries
import numpy as np
import time

import sys
sys.path.insert(1, '../../src')

from EM1D import EMf_2Lay_HVP

# Load survey details

survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()

offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Initial model

# parameters for the synthetic model
nlayer = 2 # number of layer
npos = 20 # number of sampling positions

sig_ini = [200/1000, 20/1000] # conductivities of the layers
sigmas = np.ones((npos, nlayer))*sig_ini # conductivity array

x = np.linspace(0, 20, npos) # Horizontal positions array
thk1 = 3 + x/10 +.25 # Thicknesses of first layer h_1

# Create true model in each position
model = np.stack((sigmas[:,0], sigmas[:,1], thk1), axis=1)

# Create empty array for true data in each position
data = []

# Simulate data in each position
for i in range(npos):
    data.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas[i][0], 
                             sigma2 = sigmas[i][1], 
                             h1 = thk1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt)) 
    
data = np.array(data)

# Store data and model

np.save('data/data_synth_2Lay_A2', data)
np.save('data/model_synth_2Lay_A2', model)
