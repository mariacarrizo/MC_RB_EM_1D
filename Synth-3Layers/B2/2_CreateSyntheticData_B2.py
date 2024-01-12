# Import libraries
import numpy as np
import time

import sys
sys.path.insert(1, '../../src')

from EM1D import EMf_3Lay_HVP

# Load survey details

survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()

offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Define parameters for the synthetic model
nlayer = 3 # number of layer
npos = 20 # number of sampling positions

# 3 layered conductivities
sigmas = [1/10, 1/50, 1/10] 
sigmas = np.ones((npos, nlayer))*sigmas
x = np.linspace(0, 20, npos)[:,None]

#  Thicknesses
thk1 = np.ones_like(x)*3
thk2 = np.ones_like(x) + x/10

model = np.hstack((sigmas, thk1, thk2))

# Create empty array for true data in each position
data = []

for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         sigma1 = model[pos,0],
                         sigma2 = model[pos,1],
                         sigma3 = model[pos,2],
                         h1 = model[pos,3],
                         h2 = model[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data.append(d_pos)
    
# Save model and data
np.save('data/model_synth_3Lay_B2', model)
np.save('data/data_synth_3Lay_B2', data)

# create data with noise
data_n2 = data.copy()
data_n5 = data.copy()
data_n10 = data.copy()

# error 2.5 %
np.random.seed(7)
error_2 = 0.025
data_n2 *= np.random.rand(np.size(data_n2)).reshape(np.shape(data))*error_2 +1

# error 5 %
np.random.seed(8)
error_5 = 0.05
data_n5 *= np.random.rand(np.size(data_n5)).reshape(np.shape(data))*error_5 +1

# error 10 %
np.random.seed(9)
error_10 = 0.05
data_n10 *= np.random.rand(np.size(data_n10)).reshape(np.shape(data))*error_10 +1

np.save('data/data_B2_n2', data_n2)
np.save('data/data_B2_n5', data_n5)
np.save('data/data_B2_n10', data_n10)
