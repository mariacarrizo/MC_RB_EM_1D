""" 
Author: @mariacarrizo
Date created: 15/12/2023
"""

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

sig_ini_A1_1 = [20/1000, 200/1000] # conductivities of the layers in S/m
sig_ini_A1_2 = [20/1000, 400/1000]
sig_ini_A1_3 = [20/1000, 800/1000]
sig_ini_A1_4 = [20/1000, 1600/1000]

sigmas_A1_1 = np.ones((npos, nlayer))*sig_ini_A1_1 # conductivity array
sigmas_A1_2 = np.ones((npos, nlayer))*sig_ini_A1_2 # conductivity array
sigmas_A1_3 = np.ones((npos, nlayer))*sig_ini_A1_3 # conductivity array
sigmas_A1_4 = np.ones((npos, nlayer))*sig_ini_A1_4 # conductivity array

x = np.linspace(0, 20, npos) # Horizontal positions array
h_1 = 2 + x/10 +.25 # Thicknesses of first layer h_1 in meters

# Create true model in each position
model_A1_1 = np.stack((h_1, sigmas_A1_1[:,0], sigmas_A1_1[:,1]), axis=1)
model_A1_2 = np.stack((h_1, sigmas_A1_2[:,0], sigmas_A1_2[:,1]), axis=1)
model_A1_3 = np.stack((h_1, sigmas_A1_3[:,0], sigmas_A1_3[:,1]), axis=1)
model_A1_4 = np.stack((h_1, sigmas_A1_4[:,0], sigmas_A1_4[:,1]), axis=1)

# Create empty array for true data in each position
data_A1_1 = []
data_A1_2 = []
data_A1_3 = []
data_A1_4 = []

# Simulate data in each position
for i in range(npos):
    data_A1_1.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A1_1[i][0], 
                             sigma2 = sigmas_A1_1[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt)) 
    
    data_A1_2.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A1_2[i][0], 
                             sigma2 = sigmas_A1_2[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt)) 
    
    data_A1_3.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A1_3[i][0], 
                             sigma2 = sigmas_A1_3[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt))

    data_A1_4.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A1_4[i][0], 
                             sigma2 = sigmas_A1_4[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt))

    
data_A1_1 = np.array(data_A1_1)
data_A1_2 = np.array(data_A1_2)
data_A1_3 = np.array(data_A1_3)
data_A1_4 = np.array(data_A1_4)

# Store data and model
np.save('data/data_synth_2Lay_A1_1', data_A1_1)
np.save('data/data_synth_2Lay_A1_2', data_A1_2)
np.save('data/data_synth_2Lay_A1_3', data_A1_3)
np.save('data/data_synth_2Lay_A1_4', data_A1_4)

np.save('models/model_synth_2Lay_A1_1', model_A1_1)
np.save('models/model_synth_2Lay_A1_2', model_A1_2)
np.save('models/model_synth_2Lay_A1_3', model_A1_3)
np.save('models/model_synth_2Lay_A1_4', model_A1_4)