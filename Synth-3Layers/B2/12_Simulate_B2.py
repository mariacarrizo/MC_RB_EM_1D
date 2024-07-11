#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 12_Simulate_B2.py
Description: Script to Simulate data from estimated models
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward function for 2-layered models
from EM1D import EMf_3Lay_HVP, EMf_3Lay_HVP_Q, EMf_3Lay_HVP_IP

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Load models

# Results from global search
model_GS_B2_1 = np.load('results/model_GS_B2_1.npy')
model_GS_B2_2 = np.load('results/model_GS_B2_2.npy')
model_GS_B2_3 = np.load('results/model_GS_B2_3.npy')
model_GS_B2_4 = np.load('results/model_GS_B2_4.npy')

# Results from Gauss-Newton
model_GN_B2_1 = np.load('results/model_GN_B2_1.npy')
model_GN_B2_2 = np.load('results/model_GN_B2_2.npy')
model_GN_B2_3 = np.load('results/model_GN_B2_3.npy')
model_GN_B2_4 = np.load('results/model_GN_B2_4.npy')

# Results from global search Q
model_GS_Q_B2_1 = np.load('results/model_GS_Q_B2_1.npy')
model_GS_Q_B2_2 = np.load('results/model_GS_Q_B2_2.npy')
model_GS_Q_B2_3 = np.load('results/model_GS_Q_B2_3.npy')
model_GS_Q_B2_4 = np.load('results/model_GS_Q_B2_4.npy')

# Results from Gauss-Newton Q
model_GN_Q_B2_1 = np.load('results/model_GN_Q_B2_1.npy')
model_GN_Q_B2_2 = np.load('results/model_GN_Q_B2_2.npy')
model_GN_Q_B2_3 = np.load('results/model_GN_Q_B2_3.npy')
model_GN_Q_B2_4 = np.load('results/model_GN_Q_B2_4.npy')

# Results from global search IP
model_GS_IP_B2_1 = np.load('results/model_GS_IP_B2_1.npy')
model_GS_IP_B2_2 = np.load('results/model_GS_IP_B2_2.npy')
model_GS_IP_B2_3 = np.load('results/model_GS_IP_B2_3.npy')
model_GS_IP_B2_4 = np.load('results/model_GS_IP_B2_4.npy')

# Results from Gauss-Newton IP
model_GN_IP_B2_1 = np.load('results/model_GN_IP_B2_1.npy')
model_GN_IP_B2_2 = np.load('results/model_GN_IP_B2_2.npy')
model_GN_IP_B2_3 = np.load('results/model_GN_IP_B2_3.npy')
model_GN_IP_B2_4 = np.load('results/model_GN_IP_B2_4.npy')

# Create empty array for true data in each position
data_GS_B2_1 = []
data_GS_B2_2 = []
data_GS_B2_3 = []
data_GS_B2_4 = []

data_GS_Q_B2_1 = []
data_GS_Q_B2_2 = []
data_GS_Q_B2_3 = []
data_GS_Q_B2_4 = []

data_GS_IP_B2_1 = []
data_GS_IP_B2_2 = []
data_GS_IP_B2_3 = []
data_GS_IP_B2_4 = []

data_GN_B2_1 = []
data_GN_B2_2 = []
data_GN_B2_3 = []
data_GN_B2_4 = []

data_GN_Q_B2_1 = []
data_GN_Q_B2_2 = []
data_GN_Q_B2_3 = []
data_GN_Q_B2_4 = []

data_GN_IP_B2_1 = []
data_GN_IP_B2_2 = []
data_GN_IP_B2_3 = []
data_GN_IP_B2_4 = []

# number of 1D models positions
npos = len(model_GS_B2_1)

# Simulate data in each position
for i in range(npos):
    data_GS_B2_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_B2_1[i,2], 
                                     sigma2 = model_GS_B2_1[i,3], 
                                     sigma3 = model_GS_B2_1[i,4],
                                     h1 = model_GS_B2_1[i,0], 
                                     h2 = model_GS_B2_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    data_GS_B2_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_B2_2[i,2], 
                                     sigma2 = model_GS_B2_2[i,3], 
                                     sigma3 = model_GS_B2_2[i,4],
                                     h1 = model_GS_B2_2[i,0], 
                                     h2 = model_GS_B2_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    data_GS_B2_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_B2_3[i,2], 
                                     sigma2 = model_GS_B2_3[i,3], 
                                     sigma3 = model_GS_B2_3[i,4],
                                     h1 = model_GS_B2_3[i,0], 
                                     h2 = model_GS_B2_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    data_GS_B2_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_B2_4[i,2], 
                                     sigma2 = model_GS_B2_4[i,3], 
                                     sigma3 = model_GS_B2_4[i,4],
                                     h1 = model_GS_B2_4[i,0], 
                                     h2 = model_GS_B2_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

for i in range(npos):
    data_GS_Q_B2_1.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_B2_1[i,2], 
                                     sigma2 = model_GS_Q_B2_1[i,3], 
                                     sigma3 = model_GS_Q_B2_1[i,4],
                                     h1 = model_GS_Q_B2_1[i,0], 
                                     h2 = model_GS_Q_B2_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    data_GS_Q_B2_2.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_B2_2[i,2], 
                                     sigma2 = model_GS_Q_B2_2[i,3], 
                                     sigma3 = model_GS_Q_B2_2[i,4],
                                     h1 = model_GS_Q_B2_2[i,0], 
                                     h2 = model_GS_Q_B2_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_Q_B2_3.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_B2_3[i,2], 
                                     sigma2 = model_GS_Q_B2_3[i,3], 
                                     sigma3 = model_GS_Q_B2_3[i,4],
                                     h1 = model_GS_Q_B2_3[i,0], 
                                     h2 = model_GS_Q_B2_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_Q_B2_4.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_B2_4[i,2], 
                                     sigma2 = model_GS_Q_B2_4[i,3], 
                                     sigma3 = model_GS_Q_B2_4[i,4],
                                     h1 = model_GS_Q_B2_4[i,0], 
                                     h2 = model_GS_Q_B2_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    
for i in range(npos):
    data_GS_IP_B2_1.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_B2_1[i,2], 
                                     sigma2 = model_GS_IP_B2_1[i,3], 
                                     sigma3 = model_GS_IP_B2_1[i,4],
                                     h1 = model_GS_IP_B2_1[i,0], 
                                     h2 = model_GS_IP_B2_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_IP_B2_2.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_B2_2[i,2], 
                                     sigma2 = model_GS_IP_B2_2[i,3], 
                                     sigma3 = model_GS_IP_B2_2[i,4],
                                     h1 = model_GS_IP_B2_2[i,0], 
                                     h2 = model_GS_IP_B2_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_IP_B2_3.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_B2_3[i,2], 
                                     sigma2 = model_GS_IP_B2_3[i,3], 
                                     sigma3 = model_GS_IP_B2_3[i,4],
                                     h1 = model_GS_IP_B2_3[i,0], 
                                     h2 = model_GS_IP_B2_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_IP_B2_4.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_B2_4[i,2], 
                                     sigma2 = model_GS_IP_B2_4[i,3], 
                                     sigma3 = model_GS_IP_B2_4[i,4],
                                     h1 = model_GS_IP_B2_4[i,0], 
                                     h2 = model_GS_IP_B2_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
# Simulate data in each position
for i in range(npos):
    data_GN_B2_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GN_B2_1[i,2], 
                                     sigma2 = model_GN_B2_1[i,3], 
                                     sigma3 = model_GN_B2_1[i,4],
                                     h1 = model_GN_B2_1[i,0], 
                                     h2 = model_GN_B2_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    data_GN_B2_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GN_B2_2[i,2], 
                                     sigma2 = model_GN_B2_2[i,3], 
                                     sigma3 = model_GN_B2_2[i,4],
                                     h1 = model_GN_B2_2[i,0], 
                                     h2 = model_GN_B2_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GN_B2_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GN_B2_3[i,2], 
                                     sigma2 = model_GN_B2_3[i,3], 
                                     sigma3 = model_GN_B2_3[i,4],
                                     h1 = model_GN_B2_3[i,0], 
                                     h2 = model_GN_B2_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GN_B2_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GN_B2_4[i,2], 
                                     sigma2 = model_GN_B2_4[i,3], 
                                     sigma3 = model_GN_B2_4[i,4],
                                     h1 = model_GN_B2_4[i,0], 
                                     h2 = model_GN_B2_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 


for i in range(npos):
    data_GN_Q_B2_1.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GN_Q_B2_1[i,2], 
                                     sigma2 = model_GN_Q_B2_1[i,3], 
                                     sigma3 = model_GN_Q_B2_1[i,4],
                                     h1 = model_GN_Q_B2_1[i,0], 
                                     h2 = model_GN_Q_B2_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GN_Q_B2_2.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GN_Q_B2_2[i,2], 
                                     sigma2 = model_GN_Q_B2_2[i,3], 
                                     sigma3 = model_GN_Q_B2_2[i,4],
                                     h1 = model_GN_Q_B2_2[i,0], 
                                     h2 = model_GN_Q_B2_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GN_Q_B2_3.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GN_Q_B2_3[i,2], 
                                     sigma2 = model_GN_Q_B2_3[i,3], 
                                     sigma3 = model_GN_Q_B2_3[i,4],
                                     h1 = model_GN_Q_B2_3[i,0], 
                                     h2 = model_GN_Q_B2_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GN_Q_B2_4.append(EMf_3Lay_HVP_Q(lambd, 
                                     sigma1 = model_GN_Q_B2_4[i,2], 
                                     sigma2 = model_GN_Q_B2_4[i,3], 
                                     sigma3 = model_GN_Q_B2_4[i,4],
                                     h1 = model_GN_Q_B2_4[i,0], 
                                     h2 = model_GN_Q_B2_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
for i in range(npos):
    data_GN_IP_B2_1.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GN_IP_B2_1[i,2], 
                                     sigma2 = model_GN_IP_B2_1[i,3], 
                                     sigma3 = model_GN_IP_B2_1[i,4],
                                     h1 = model_GN_IP_B2_1[i,0], 
                                     h2 = model_GN_IP_B2_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GN_IP_B2_2.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GN_IP_B2_2[i,2], 
                                     sigma2 = model_GN_IP_B2_2[i,3], 
                                     sigma3 = model_GN_IP_B2_2[i,4],
                                     h1 = model_GN_IP_B2_2[i,0], 
                                     h2 = model_GN_IP_B2_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))  
    
    data_GN_IP_B2_3.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GN_IP_B2_3[i,2], 
                                     sigma2 = model_GN_IP_B2_3[i,3], 
                                     sigma3 = model_GN_IP_B2_3[i,4],
                                     h1 = model_GN_IP_B2_3[i,0], 
                                     h2 = model_GN_IP_B2_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    data_GN_IP_B2_4.append(EMf_3Lay_HVP_IP(lambd, 
                                     sigma1 = model_GN_IP_B2_4[i,2], 
                                     sigma2 = model_GN_IP_B2_4[i,3], 
                                     sigma3 = model_GN_IP_B2_4[i,4],
                                     h1 = model_GN_IP_B2_4[i,0], 
                                     h2 = model_GN_IP_B2_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 


# Store data and model
np.save('data/data_GS_B2_1', data_GS_B2_1)
np.save('data/data_GS_B2_2', data_GS_B2_2)
np.save('data/data_GS_B2_3', data_GS_B2_3)
np.save('data/data_GS_B2_4', data_GS_B2_4)

np.save('data/data_GS_Q_B2_1', data_GS_Q_B2_1)
np.save('data/data_GS_Q_B2_2', data_GS_Q_B2_2)
np.save('data/data_GS_Q_B2_3', data_GS_Q_B2_3)
np.save('data/data_GS_Q_B2_4', data_GS_Q_B2_4)

np.save('data/data_GS_IP_B2_1', data_GS_IP_B2_1)
np.save('data/data_GS_IP_B2_2', data_GS_IP_B2_2)
np.save('data/data_GS_IP_B2_3', data_GS_IP_B2_3)
np.save('data/data_GS_IP_B2_4', data_GS_IP_B2_4)

np.save('data/data_GN_B2_1', data_GN_B2_1)
np.save('data/data_GN_B2_2', data_GN_B2_2)
np.save('data/data_GN_B2_3', data_GN_B2_3)
np.save('data/data_GN_B2_4', data_GN_B2_4)

np.save('data/data_GN_Q_B2_1', data_GN_Q_B2_1)
np.save('data/data_GN_Q_B2_2', data_GN_Q_B2_2)
np.save('data/data_GN_Q_B2_3', data_GN_Q_B2_3)
np.save('data/data_GN_Q_B2_4', data_GN_Q_B2_4)

np.save('data/data_GN_IP_B2_1', data_GN_IP_B2_1)
np.save('data/data_GN_IP_B2_2', data_GN_IP_B2_2)
np.save('data/data_GN_IP_B2_3', data_GN_IP_B2_3)
np.save('data/data_GN_IP_B2_4', data_GN_IP_B2_4)