# Simulate data from estimated models

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward function for 2-layered models
from EM1D import EMf_3Lay_HVP

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Load models

# Results from global search
model_GS_n2_B1_1 = np.load('results/model_GS_n2_B1_1.npy')
model_GS_n2_B1_2 = np.load('results/model_GS_n2_B1_2.npy')
model_GS_n2_B1_3 = np.load('results/model_GS_n2_B1_3.npy')
model_GS_n2_B1_4 = np.load('results/model_GS_n2_B1_4.npy')

model_GS_n5_B1_1 = np.load('results/model_GS_n5_B1_1.npy')
model_GS_n5_B1_2 = np.load('results/model_GS_n5_B1_2.npy')
model_GS_n5_B1_3 = np.load('results/model_GS_n5_B1_3.npy')
model_GS_n5_B1_4 = np.load('results/model_GS_n5_B1_4.npy')

model_GS_n10_B1_1 = np.load('results/model_GS_n10_B1_1.npy')
model_GS_n10_B1_2 = np.load('results/model_GS_n10_B1_2.npy')
model_GS_n10_B1_3 = np.load('results/model_GS_n10_B1_3.npy')
model_GS_n10_B1_4 = np.load('results/model_GS_n10_B1_4.npy')

model_Opt_n2_B1_1 = np.load('results/model_Opt_n2_B1_1.npy')
model_Opt_n2_B1_2 = np.load('results/model_Opt_n2_B1_2.npy')
model_Opt_n2_B1_3 = np.load('results/model_Opt_n2_B1_3.npy')
model_Opt_n2_B1_4 = np.load('results/model_Opt_n2_B1_4.npy')

model_Opt_n5_B1_1 = np.load('results/model_Opt_n5_B1_1.npy')
model_Opt_n5_B1_2 = np.load('results/model_Opt_n5_B1_2.npy')
model_Opt_n5_B1_3 = np.load('results/model_Opt_n5_B1_3.npy')
model_Opt_n5_B1_4 = np.load('results/model_Opt_n5_B1_4.npy')

model_Opt_n10_B1_1 = np.load('results/model_Opt_n10_B1_1.npy')
model_Opt_n10_B1_2 = np.load('results/model_Opt_n10_B1_2.npy')
model_Opt_n10_B1_3 = np.load('results/model_Opt_n10_B1_3.npy')
model_Opt_n10_B1_4 = np.load('results/model_Opt_n10_B1_4.npy')

# Create empty array for true data in each position
data_GS_n2_B1_1 = []
data_GS_n2_B1_2 = []
data_GS_n2_B1_3 = []
data_GS_n2_B1_4 = []

data_GS_n5_B1_1 = []
data_GS_n5_B1_2 = []
data_GS_n5_B1_3 = []
data_GS_n5_B1_4 = []

data_GS_n10_B1_1 = []
data_GS_n10_B1_2 = []
data_GS_n10_B1_3 = []
data_GS_n10_B1_4 = []

data_Opt_n2_B1_1 = []
data_Opt_n2_B1_2 = []
data_Opt_n2_B1_3 = []
data_Opt_n2_B1_4 = []

data_Opt_n5_B1_1 = []
data_Opt_n5_B1_2 = []
data_Opt_n5_B1_3 = []
data_Opt_n5_B1_4 = []

data_Opt_n10_B1_1 = []
data_Opt_n10_B1_2 = []
data_Opt_n10_B1_3 = []
data_Opt_n10_B1_4 = []


npos = len(model_GS_n2_B1_1)

# Simulate data in each position
for i in range(npos):
    data_GS_n2_B1_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n2_B1_1[i,2], 
                                     sigma2 = model_GS_n2_B1_1[i,3],
                                     sigma3 = model_GS_n2_B1_1[i,4],
                                     h1 = model_GS_n2_B1_1[i,0], 
                                     h2 = model_GS_n2_B1_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n2_B1_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n2_B1_2[i,2], 
                                     sigma2 = model_GS_n2_B1_2[i,3], 
                                     sigma3 = model_GS_n2_B1_2[i,4],
                                     h1 = model_GS_n2_B1_2[i,0], 
                                     h2 = model_GS_n2_B1_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n2_B1_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n2_B1_3[i,2], 
                                     sigma2 = model_GS_n2_B1_3[i,3], 
                                     sigma3 = model_GS_n2_B1_3[i,4],
                                     h1 = model_GS_n2_B1_3[i,0], 
                                     h2 = model_GS_n2_B1_3[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n2_B1_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n2_B1_4[i,2], 
                                     sigma2 = model_GS_n2_B1_4[i,3], 
                                     sigma3 = model_GS_n2_B1_4[i,4],
                                     h1 = model_GS_n2_B1_4[i,0], 
                                     h2 = model_GS_n2_B1_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

for i in range(npos):
    data_GS_n5_B1_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n5_B1_1[i,2], 
                                     sigma2 = model_GS_n5_B1_1[i,3], 
                                     sigma3 = model_GS_n5_B1_1[i,4],
                                     h1 = model_GS_n5_B1_1[i,0], 
                                     h2 = model_GS_n5_B1_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n5_B1_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n5_B1_2[i,2], 
                                     sigma2 = model_GS_n5_B1_2[i,3], 
                                     sigma3 = model_GS_n5_B1_2[i,4],
                                     h1 = model_GS_n5_B1_2[i,0], 
                                     h2 = model_GS_n5_B1_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n5_B1_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n5_B1_3[i,2], 
                                     sigma2 = model_GS_n5_B1_3[i,3], 
                                     sigma3 = model_GS_n5_B1_3[i,4],
                                     h1 = model_GS_n5_B1_3[i,0], 
                                     h2 = model_GS_n5_B1_3[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n5_B1_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n5_B1_4[i,2], 
                                     sigma2 = model_GS_n5_B1_4[i,3], 
                                     sigma3 = model_GS_n5_B1_4[i,4],
                                     h1 = model_GS_n5_B1_4[i,0], 
                                     h2 = model_GS_n5_B1_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
for i in range(npos):
    data_GS_n10_B1_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n10_B1_1[i,2], 
                                     sigma2 = model_GS_n10_B1_1[i,3], 
                                     sigma3 = model_GS_n10_B1_1[i,4],
                                     h1 = model_GS_n10_B1_1[i,0], 
                                     h2 = model_GS_n10_B1_1[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n10_B1_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n10_B1_2[i,2], 
                                     sigma2 = model_GS_n10_B1_2[i,3], 
                                     sigma3 = model_GS_n10_B1_2[i,4],
                                     h1 = model_GS_n10_B1_2[i,0], 
                                     h2 = model_GS_n10_B1_2[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n10_B1_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n10_B1_3[i,2], 
                                     sigma2 = model_GS_n10_B1_3[i,3], 
                                     sigma3 = model_GS_n10_B1_3[i,4],
                                     h1 = model_GS_n10_B1_3[i,0], 
                                     h2 = model_GS_n10_B1_3[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_GS_n10_B1_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_GS_n10_B1_4[i,2], 
                                     sigma2 = model_GS_n10_B1_4[i,3], 
                                     sigma3 = model_GS_n10_B1_4[i,4],
                                     h1 = model_GS_n10_B1_4[i,0], 
                                     h2 = model_GS_n10_B1_4[i,1],
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

    
# Simulate data in each position
for i in range(npos):
    data_Opt_n2_B1_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n2_B1_1[i,2], 
                                     sigma2 = model_Opt_n2_B1_1[i,3], 
                                     sigma3 = model_Opt_n2_B1_1[i,4],
                                     h1 = model_Opt_n2_B1_1[i,0], 
                                     h2 = model_Opt_n2_B1_1[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n2_B1_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n2_B1_2[i,2], 
                                     sigma2 = model_Opt_n2_B1_2[i,3], 
                                     sigma3 = model_Opt_n2_B1_2[i,4],
                                     h1 = model_Opt_n2_B1_2[i,0], 
                                     h2 = model_Opt_n2_B1_2[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n2_B1_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n2_B1_3[i,2], 
                                     sigma2 = model_Opt_n2_B1_3[i,3], 
                                     sigma3 = model_Opt_n2_B1_3[i,4],
                                     h1 = model_Opt_n2_B1_3[i,0], 
                                     h2 = model_Opt_n2_B1_3[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n2_B1_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n2_B1_4[i,2], 
                                     sigma2 = model_Opt_n2_B1_4[i,3], 
                                     sigma3 = model_Opt_n2_B1_4[i,4],
                                     h1 = model_Opt_n2_B1_4[i,0], 
                                     h2 = model_Opt_n2_B1_4[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

for i in range(npos):
    data_Opt_n5_B1_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n5_B1_1[i,2], 
                                     sigma2 = model_Opt_n5_B1_1[i,3], 
                                     sigma3 = model_Opt_n5_B1_1[i,4],
                                     h1 = model_Opt_n5_B1_1[i,0], 
                                     h2 = model_Opt_n5_B1_1[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n5_B1_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n5_B1_2[i,2], 
                                     sigma2 = model_Opt_n5_B1_2[i,3], 
                                     sigma3 = model_Opt_n5_B1_2[i,4],
                                     h1 = model_Opt_n5_B1_2[i,0], 
                                     h2 = model_Opt_n5_B1_2[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n5_B1_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n5_B1_3[i,2], 
                                     sigma2 = model_Opt_n5_B1_3[i,3], 
                                     sigma3 = model_Opt_n5_B1_3[i,4],
                                     h1 = model_Opt_n5_B1_3[i,0], 
                                     h2 = model_Opt_n5_B1_3[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n5_B1_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n5_B1_4[i,2], 
                                     sigma2 = model_Opt_n5_B1_4[i,3], 
                                     sigma3 = model_Opt_n5_B1_4[i,4],
                                     h1 = model_Opt_n5_B1_4[i,0], 
                                     h2 = model_Opt_n5_B1_4[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 

for i in range(npos):
    data_Opt_n10_B1_1.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n10_B1_1[i,2], 
                                     sigma2 = model_Opt_n10_B1_1[i,3], 
                                     sigma3 = model_Opt_n10_B1_1[i,4],
                                     h1 = model_Opt_n10_B1_1[i,0], 
                                     h2 = model_Opt_n10_B1_1[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n10_B1_2.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n10_B1_2[i,2], 
                                     sigma2 = model_Opt_n10_B1_2[i,3], 
                                     sigma3 = model_Opt_n10_B1_2[i,4],
                                     h1 = model_Opt_n10_B1_2[i,0], 
                                     h2 = model_Opt_n10_B1_2[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n10_B1_3.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n10_B1_3[i,2], 
                                     sigma2 = model_Opt_n10_B1_3[i,3], 
                                     sigma3 = model_Opt_n10_B1_3[i,4],
                                     h1 = model_Opt_n10_B1_3[i,0], 
                                     h2 = model_Opt_n10_B1_3[i,1],  
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    data_Opt_n10_B1_4.append(EMf_3Lay_HVP(lambd, 
                                     sigma1 = model_Opt_n10_B1_4[i,2], 
                                     sigma2 = model_Opt_n10_B1_4[i,3], 
                                     sigma3 = model_Opt_n10_B1_4[i,4],
                                     h1 = model_Opt_n10_B1_4[i,0], 
                                     h2 = model_Opt_n10_B1_4[i,1], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 


# Store data and model
np.save('data/data_GS_n2_B1_1', data_GS_n2_B1_1)
np.save('data/data_GS_n2_B1_2', data_GS_n2_B1_2)
np.save('data/data_GS_n2_B1_3', data_GS_n2_B1_3)
np.save('data/data_GS_n2_B1_4', data_GS_n2_B1_4)

np.save('data/data_GS_n5_B1_1', data_GS_n5_B1_1)
np.save('data/data_GS_n5_B1_2', data_GS_n5_B1_2)
np.save('data/data_GS_n5_B1_3', data_GS_n5_B1_3)
np.save('data/data_GS_n5_B1_4', data_GS_n5_B1_4)

np.save('data/data_GS_n10_B1_1', data_GS_n10_B1_1)
np.save('data/data_GS_n10_B1_2', data_GS_n10_B1_2)
np.save('data/data_GS_n10_B1_3', data_GS_n10_B1_3)
np.save('data/data_GS_n10_B1_4', data_GS_n10_B1_4)

np.save('data/data_Opt_n2_B1_1', data_Opt_n2_B1_1)
np.save('data/data_Opt_n2_B1_2', data_Opt_n2_B1_2)
np.save('data/data_Opt_n2_B1_3', data_Opt_n2_B1_3)
np.save('data/data_Opt_n2_B1_4', data_Opt_n2_B1_4)

np.save('data/data_Opt_n5_B1_1', data_Opt_n5_B1_1)
np.save('data/data_Opt_n5_B1_2', data_Opt_n5_B1_2)
np.save('data/data_Opt_n5_B1_3', data_Opt_n5_B1_3)
np.save('data/data_Opt_n5_B1_4', data_Opt_n5_B1_4)

np.save('data/data_Opt_n10_B1_1', data_Opt_n10_B1_1)
np.save('data/data_Opt_n10_B1_2', data_Opt_n10_B1_2)
np.save('data/data_Opt_n10_B1_3', data_Opt_n10_B1_3)
np.save('data/data_Opt_n10_B1_4', data_Opt_n10_B1_4)

