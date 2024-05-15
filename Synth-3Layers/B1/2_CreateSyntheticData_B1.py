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
sig_ini_B1_1 = [20/1000, 200/1000, 20/1000] # in S/m
sig_ini_B1_2 = [20/1000, 400/1000, 20/1000]
sig_ini_B1_3 = [20/1000, 800/1000, 20/1000]
sig_ini_B1_4 = [20/1000, 1600/1000, 20/1000]

sigmas_B1_1 = np.ones((npos, nlayer))*sig_ini_B1_1
sigmas_B1_2 = np.ones((npos, nlayer))*sig_ini_B1_2
sigmas_B1_3 = np.ones((npos, nlayer))*sig_ini_B1_3
sigmas_B1_4 = np.ones((npos, nlayer))*sig_ini_B1_4

x = np.linspace(0, 20, npos)[:,None]

#  Thicknesses in m
thk1 = np.ones_like(x)*3
thk2 = np.ones_like(x) + x/10

model_B1_1 = np.hstack((thk1, thk2, sigmas_B1_1))
model_B1_2 = np.hstack((thk1, thk2, sigmas_B1_2))
model_B1_3 = np.hstack((thk1, thk2, sigmas_B1_3))
model_B1_4 = np.hstack((thk1, thk2, sigmas_B1_4))

# Create empty array for true data in each position
data_B1_1 = []
data_B1_2 = []
data_B1_3 = []
data_B1_4 = []

for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B1_1[pos,0],
                         h2 = model_B1_1[pos,1],
                         sigma1 = model_B1_1[pos,2],
                         sigma2 = model_B1_1[pos,3],
                         sigma3 = model_B1_1[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B1_1.append(d_pos)
    
for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B1_2[pos,0],
                         h2 = model_B1_2[pos,1],
                         sigma1 = model_B1_2[pos,2],
                         sigma2 = model_B1_2[pos,3],
                         sigma3 = model_B1_2[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B1_2.append(d_pos)
    
for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B1_3[pos,0],
                         h2 = model_B1_3[pos,1],
                         sigma1 = model_B1_3[pos,2],
                         sigma2 = model_B1_3[pos,3],
                         sigma3 = model_B1_3[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B1_3.append(d_pos)
    
for pos in range(npos):
    d_pos = EMf_3Lay_HVP(lambd,
                         h1 = model_B1_4[pos,0],
                         h2 = model_B1_4[pos,1],
                         sigma1 = model_B1_4[pos,2],
                         sigma2 = model_B1_4[pos,3],
                         sigma3 = model_B1_4[pos,4],
                         height = height,
                         offsets = offsets,
                         freq = freq,
                         filt = filt)
    data_B1_4.append(d_pos)
    
# Save model and data
np.save('models/model_synth_B1_1', model_B1_1)
np.save('models/model_synth_B1_2', model_B1_2)
np.save('models/model_synth_B1_3', model_B1_3)
np.save('models/model_synth_B1_4', model_B1_4)

np.save('data/data_synth_B1_1', data_B1_1)
np.save('data/data_synth_B1_2', data_B1_2)
np.save('data/data_synth_B1_3', data_B1_3)
np.save('data/data_synth_B1_4', data_B1_4)

# create data with noise
data_noise_2_B1_1 = data_B1_1.copy()
data_noise_5_B1_1 = data_B1_1.copy()
data_noise_10_B1_1 = data_B1_1.copy()

data_noise_2_B1_2 = data_B1_2.copy()
data_noise_5_B1_2 = data_B1_2.copy()
data_noise_10_B1_2 = data_B1_2.copy()

data_noise_2_B1_3 = data_B1_3.copy()
data_noise_5_B1_3 = data_B1_3.copy()
data_noise_10_B1_3 = data_B1_3.copy()

data_noise_2_B1_4 = data_B1_4.copy()
data_noise_5_B1_4 = data_B1_4.copy()
data_noise_10_B1_4 = data_B1_4.copy()

# for noise 2.5 %
error_2 = 0.025
np.random.seed(13) 
data_noise_2_B1_1 *= np.random.randn(np.size(data_B1_1)).reshape(np.shape(data_B1_1))* error_2+1

np.random.seed(14) 
data_noise_2_B1_2 *= np.random.randn(np.size(data_B1_2)).reshape(np.shape(data_B1_2))* error_2+1

np.random.seed(15) 
data_noise_2_B1_3 *= np.random.randn(np.size(data_B1_3)).reshape(np.shape(data_B1_3))* error_2+1

np.random.seed(16) 
data_noise_2_B1_4 *= np.random.randn(np.size(data_B1_4)).reshape(np.shape(data_B1_4))* error_2+1

# for noise 5 %
error_5 = 0.05
np.random.seed(17) 
data_noise_5_B1_1 *= np.random.randn(np.size(data_B1_1)).reshape(np.shape(data_B1_1))* error_5+1

np.random.seed(18) 
data_noise_5_B1_2 *= np.random.randn(np.size(data_B1_2)).reshape(np.shape(data_B1_2))* error_5+1

np.random.seed(19) 
data_noise_5_B1_3 *= np.random.randn(np.size(data_B1_3)).reshape(np.shape(data_B1_3))* error_5+1

np.random.seed(20) 
data_noise_5_B1_4 *= np.random.randn(np.size(data_B1_4)).reshape(np.shape(data_B1_4))* error_5+1

# for noise 10%
error_10 = 0.1

np.random.seed(21)
data_noise_10_B1_1 *= np.random.randn(np.size(data_B1_1)).reshape(np.shape(data_B1_1))* error_10+1

np.random.seed(22)
data_noise_10_B1_2 *= np.random.randn(np.size(data_B1_2)).reshape(np.shape(data_B1_2))* error_10+1

np.random.seed(23)
data_noise_10_B1_3 *= np.random.randn(np.size(data_B1_3)).reshape(np.shape(data_B1_3))* error_10+1

np.random.seed(24)
data_noise_10_B1_4 *= np.random.randn(np.size(data_B1_4)).reshape(np.shape(data_B1_4))* error_10+1

# Save noisy data
np.save('data/data_n2_B1_1', data_noise_2_B1_1)
np.save('data/data_n2_B1_2', data_noise_2_B1_2)
np.save('data/data_n2_B1_3', data_noise_2_B1_3)
np.save('data/data_n2_B1_4', data_noise_2_B1_4)

np.save('data/data_n5_B1_1', data_noise_5_B1_1)
np.save('data/data_n5_B1_2', data_noise_5_B1_2)
np.save('data/data_n5_B1_3', data_noise_5_B1_3)
np.save('data/data_n5_B1_4', data_noise_5_B1_4)

np.save('data/data_n10_B1_1', data_noise_10_B1_1)
np.save('data/data_n10_B1_2', data_noise_10_B1_2)
np.save('data/data_n10_B1_3', data_noise_10_B1_3)
np.save('data/data_n10_B1_4', data_noise_10_B1_4)
