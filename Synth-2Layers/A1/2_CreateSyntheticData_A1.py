# File to create synthetic models in case A.1

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward function for 2-layered models
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
npos = 20 # number of 1D models

sig_ini = [20/1000, 200/1000] # conductivities of the layers in S/m
sigmas = np.ones((npos, nlayer))*sig_ini # conductivity array

x = np.linspace(0, 20, npos) # Horizontal positions array
h_1 = 3 + x/10 +.25 # Thicknesses of first layer h_1 in meters

# Create true model in each position
model = np.stack((sigmas[:,0], sigmas[:,1], h_1), axis=1)

# Create empty array for true data in each position
data = []

# Simulate data in each position
for i in range(npos):
    data.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas[i][0], 
                             sigma2 = sigmas[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt)) 
    
data = np.array(data)

# Store data and model
np.save('data/data_synth_2Lay_A1', data)
np.save('data/model_synth_2Lay_A1', model)