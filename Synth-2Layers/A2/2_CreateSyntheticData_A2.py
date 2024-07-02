#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 2_CreateSyntheticData_A2.py
Description: Script to create synthetic models in A2 cases
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 16/12/2023
"""

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

sig_ini_A2_1 = [200/1000, 20/1000] # conductivities of the layers in S/m
sig_ini_A2_2 = [400/1000, 20/1000]
sig_ini_A2_3 = [800/1000, 20/1000]
sig_ini_A2_4 = [1600/1000, 20/1000]

sigmas_A2_1 = np.ones((npos, nlayer))*sig_ini_A2_1 # conductivity array
sigmas_A2_2 = np.ones((npos, nlayer))*sig_ini_A2_2 # conductivity array
sigmas_A2_3 = np.ones((npos, nlayer))*sig_ini_A2_3 # conductivity array
sigmas_A2_4 = np.ones((npos, nlayer))*sig_ini_A2_4 # conductivity array

x = np.linspace(0, 20, npos) # Horizontal positions array
h_1 = 2 + x/10 +.25 # Thicknesses of first layer h_1 in meters

# Create true model in each position
model_A2_1 = np.stack((h_1, sigmas_A2_1[:,0], sigmas_A2_1[:,1]), axis=1)
model_A2_2 = np.stack((h_1, sigmas_A2_2[:,0], sigmas_A2_2[:,1]), axis=1)
model_A2_3 = np.stack((h_1, sigmas_A2_3[:,0], sigmas_A2_3[:,1]), axis=1)
model_A2_4 = np.stack((h_1, sigmas_A2_4[:,0], sigmas_A2_4[:,1]), axis=1)

# Create empty array for true data in each position
data_A2_1 = []
data_A2_2 = []
data_A2_3 = []
data_A2_4 = []

# Simulate data in each position
for i in range(npos):
    data_A2_1.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A2_1[i][0], 
                             sigma2 = sigmas_A2_1[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt)) 
    
    data_A2_2.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A2_2[i][0], 
                             sigma2 = sigmas_A2_2[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt)) 
    
    data_A2_3.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A2_3[i][0], 
                             sigma2 = sigmas_A2_3[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt))

    data_A2_4.append(EMf_2Lay_HVP(lambd, 
                             sigma1 = sigmas_A2_4[i][0], 
                             sigma2 = sigmas_A2_4[i][1], 
                             h1 = h_1[i], 
                             height = height,
                             offsets = offsets,
                             freq = freq,
                             filt = filt))

# Store data and models
np.save('data/data_synth_2Lay_A2_1', data_A2_1)
np.save('data/data_synth_2Lay_A2_2', data_A2_2)
np.save('data/data_synth_2Lay_A2_3', data_A2_3)
np.save('data/data_synth_2Lay_A2_4', data_A2_4)

np.save('models/model_synth_2Lay_A2_1', model_A2_1)
np.save('models/model_synth_2Lay_A2_2', model_A2_2)
np.save('models/model_synth_2Lay_A2_3', model_A2_3)
np.save('models/model_synth_2Lay_A2_4', model_A2_4)