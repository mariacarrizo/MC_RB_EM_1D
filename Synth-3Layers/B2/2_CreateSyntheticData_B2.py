# Script to simulate synthetic data for case B.1

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward function for 3-layered 1D models
from EM1D import EMf_3Lay_HVP

# Load survey details

survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()

offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Define parameters for the synthetic model
nlayer = 3 # number of layers
npos = 20 # number of sampling positions

# 3 layered conductivities
sig_ini_B2_1 = [200/1000, 20/1000, 200/1000] # in S/m
sig_ini_B2_2 = [400/1000, 20/1000, 400/1000]
sig_ini_B2_3 = [800/1000, 20/1000, 800/1000]
sig_ini_B2_4 = [1600/1000, 20/1000, 1600/1000]

sigmas_B2_1 = np.ones((npos, nlayer))*sig_ini_B2_1
sigmas_B2_2 = np.ones((npos, nlayer))*sig_ini_B2_2
sigmas_B2_3 = np.ones((npos, nlayer))*sig_ini_B2_3
sigmas_B2_4 = np.ones((npos, nlayer))*sig_ini_B2_4

x = np.linspace(0, 20, npos)[:,None]

#  Thicknesses in m
thk1 = np.ones_like(x)*3
thk2 = np.ones_like(x) + x/10

model_B2_1 = np.hstack((thk1, thk2, sigmas_B2_1))
model_B2_2 = np.hstack((thk1, thk2, sigmas_B2_2))
model_B2_3 = np.hstack((thk1, thk2, sigmas_B2_3))
model_B2_4 = np.hstack((thk1, thk2, sigmas_B2_4))

# Create empty array for true data in each position
data_B2_1 = []
data_B2_2 = []
data_B2_3 = []
data_B2_4 = []

for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B2_1[pos,0],
                         h2 = model_B2_1[pos,1],
                         sigma1 = model_B2_1[pos,2],
                         sigma2 = model_B2_1[pos,3],
                         sigma3 = model_B2_1[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B2_1.append(d_pos)
    
for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B2_2[pos,0],
                         h2 = model_B2_2[pos,1],
                         sigma1 = model_B2_2[pos,2],
                         sigma2 = model_B2_2[pos,3],
                         sigma3 = model_B2_2[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B2_2.append(d_pos)
    
for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B2_3[pos,0],
                         h2 = model_B2_3[pos,1],
                         sigma1 = model_B2_3[pos,2],
                         sigma2 = model_B2_3[pos,3],
                         sigma3 = model_B2_3[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B2_3.append(d_pos)
    
for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B2_4[pos,0],
                         h2 = model_B2_4[pos,1],
                         sigma1 = model_B2_4[pos,2],
                         sigma2 = model_B2_4[pos,3],
                         sigma3 = model_B2_4[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B2_4.append(d_pos)
    
# Save model and data
np.save('models/model_synth_B2_1', model_B2_1)
np.save('models/model_synth_B2_2', model_B2_2)
np.save('models/model_synth_B2_3', model_B2_3)
np.save('models/model_synth_B2_4', model_B2_4)

np.save('data/data_synth_B2_1', data_B2_1)
np.save('data/data_synth_B2_2', data_B2_2)
np.save('data/data_synth_B2_3', data_B2_3)
np.save('data/data_synth_B2_4', data_B2_4)

# create data with noise
data_noise_2_B2_1 = data_B2_1.copy()
data_noise_5_B2_1 = data_B2_1.copy()
data_noise_10_B2_1 = data_B2_1.copy()

data_noise_2_B2_2 = data_B2_2.copy()
data_noise_5_B2_2 = data_B2_2.copy()
data_noise_10_B2_2 = data_B2_2.copy()

data_noise_2_B2_3 = data_B2_3.copy()
data_noise_5_B2_3 = data_B2_3.copy()
data_noise_10_B2_3 = data_B2_3.copy()

data_noise_2_B2_4 = data_B2_4.copy()
data_noise_5_B2_4 = data_B2_4.copy()
data_noise_10_B2_4 = data_B2_4.copy()

# for noise 2.5 %
error_2 = 0.025
np.random.seed(25) 
data_noise_2_B2_1 *= np.random.randn(np.size(data_B2_1)).reshape(np.shape(data_B2_1))* error_2+1

np.random.seed(26) 
data_noise_2_B2_2 *= np.random.randn(np.size(data_B2_2)).reshape(np.shape(data_B2_2))* error_2+1

np.random.seed(27) 
data_noise_2_B2_3 *= np.random.randn(np.size(data_B2_3)).reshape(np.shape(data_B2_3))* error_2+1

np.random.seed(28) 
data_noise_2_B2_4 *= np.random.randn(np.size(data_B2_4)).reshape(np.shape(data_B2_4))* error_2+1

# for noise 5 %
error_5 = 0.05
np.random.seed(29) 
data_noise_5_B2_1 *= np.random.randn(np.size(data_B2_1)).reshape(np.shape(data_B2_1))* error_5+1

np.random.seed(30) 
data_noise_5_B2_2 *= np.random.randn(np.size(data_B2_2)).reshape(np.shape(data_B2_2))* error_5+1

np.random.seed(31) 
data_noise_5_B2_3 *= np.random.randn(np.size(data_B2_3)).reshape(np.shape(data_B2_3))* error_5+1

np.random.seed(32) 
data_noise_5_B2_4 *= np.random.randn(np.size(data_B2_4)).reshape(np.shape(data_B2_4))* error_5+1

# for noise 10%
error_10 = 0.1

np.random.seed(33)
data_noise_10_B2_1 *= np.random.randn(np.size(data_B2_1)).reshape(np.shape(data_B2_1))* error_10+1

np.random.seed(34)
data_noise_10_B2_2 *= np.random.randn(np.size(data_B2_2)).reshape(np.shape(data_B2_2))* error_10+1

np.random.seed(35)
data_noise_10_B2_3 *= np.random.randn(np.size(data_B2_3)).reshape(np.shape(data_B2_3))* error_10+1

np.random.seed(36)
data_noise_10_B2_4 *= np.random.randn(np.size(data_B2_4)).reshape(np.shape(data_B2_4))* error_10+1

# Save noisy data
np.save('data/data_n2_B2_1', data_noise_2_B2_1)
np.save('data/data_n2_B2_2', data_noise_2_B2_2)
np.save('data/data_n2_B2_3', data_noise_2_B2_3)
np.save('data/data_n2_B2_4', data_noise_2_B2_4)

np.save('data/data_n5_B2_1', data_noise_5_B2_1)
np.save('data/data_n5_B2_2', data_noise_5_B2_2)
np.save('data/data_n5_B2_3', data_noise_5_B2_3)
np.save('data/data_n5_B2_4', data_noise_5_B2_4)

np.save('data/data_n10_B2_1', data_noise_10_B2_1)
np.save('data/data_n10_B2_2', data_noise_10_B2_2)
np.save('data/data_n10_B2_3', data_noise_10_B2_3)
np.save('data/data_n10_B2_4', data_noise_10_B2_4)
