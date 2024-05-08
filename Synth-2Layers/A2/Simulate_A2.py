# Simulate data from estimated models

# Import libraries
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward function for 2-layered models
from EM1D import EMf_2Lay_HVP, EMf_2Lay_HVP_Q, EMf_2Lay_HVP_IP

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Load models

# Results from global search
model_GS_A2_1 = np.load('results/model_2Lay_A2_1.npy')
model_GS_A2_2 = np.load('results/model_2Lay_A2_2.npy')
model_GS_A2_3 = np.load('results/model_2Lay_A2_3.npy')
model_GS_A2_4 = np.load('results/model_2Lay_A2_4.npy')

# Results from optimization
model_Opt_A2_1 = np.load('results/model_Opt_A2_1.npy')
model_Opt_A2_2 = np.load('results/model_Opt_A2_2.npy')
model_Opt_A2_3 = np.load('results/model_Opt_A2_3.npy')
model_Opt_A2_4 = np.load('results/model_Opt_A2_4.npy')

# Results from global search Q
model_GS_Q_A2_1 = np.load('results/model_2Lay_Q_A2_1.npy')
model_GS_Q_A2_2 = np.load('results/model_2Lay_Q_A2_2.npy')
model_GS_Q_A2_3 = np.load('results/model_2Lay_Q_A2_3.npy')
model_GS_Q_A2_4 = np.load('results/model_2Lay_Q_A2_4.npy')

# Results from optimization Q
model_Opt_Q_A2_1 = np.load('results/model_Opt_Q_A2_1.npy')
model_Opt_Q_A2_2 = np.load('results/model_Opt_Q_A2_2.npy')
model_Opt_Q_A2_3 = np.load('results/model_Opt_Q_A2_3.npy')
model_Opt_Q_A2_4 = np.load('results/model_Opt_Q_A2_4.npy')

# Results from global search IP
model_GS_IP_A2_1 = np.load('results/model_2Lay_IP_A2_1.npy')
model_GS_IP_A2_2 = np.load('results/model_2Lay_IP_A2_2.npy')
model_GS_IP_A2_3 = np.load('results/model_2Lay_IP_A2_3.npy')
model_GS_IP_A2_4 = np.load('results/model_2Lay_IP_A2_4.npy')

# Results from optimization IP
model_Opt_IP_A2_1 = np.load('results/model_Opt_IP_A2_1.npy')
model_Opt_IP_A2_2 = np.load('results/model_Opt_IP_A2_2.npy')
model_Opt_IP_A2_3 = np.load('results/model_Opt_IP_A2_3.npy')
model_Opt_IP_A2_4 = np.load('results/model_Opt_IP_A2_4.npy')

# Create empty array for true data in each position
data_GS_A2_1 = []
data_GS_A2_2 = []
data_GS_A2_3 = []
data_GS_A2_4 = []

data_GS_Q_A2_1 = []
data_GS_Q_A2_2 = []
data_GS_Q_A2_3 = []
data_GS_Q_A2_4 = []

data_GS_IP_A2_1 = []
data_GS_IP_A2_2 = []
data_GS_IP_A2_3 = []
data_GS_IP_A2_4 = []

data_Opt_A2_1 = []
data_Opt_A2_2 = []
data_Opt_A2_3 = []
data_Opt_A2_4 = []

data_Opt_Q_A2_1 = []
data_Opt_Q_A2_2 = []
data_Opt_Q_A2_3 = []
data_Opt_Q_A2_4 = []

data_Opt_IP_A2_1 = []
data_Opt_IP_A2_2 = []
data_Opt_IP_A2_3 = []
data_Opt_IP_A2_4 = []

npos = len(model_GS_A2_1)

# Simulate data in each position
for i in range(npos):
    data_GS_A2_1.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_GS_A2_1[i,1], 
                                     sigma2 = model_GS_A2_1[i,2], 
                                     h1 = model_GS_A2_1[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_A2_2.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_GS_A2_2[i,1], 
                                     sigma2 = model_GS_A2_2[i,2], 
                                     h1 = model_GS_A2_2[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_A2_3.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_GS_A2_3[i,1], 
                                     sigma2 = model_GS_A2_3[i,2], 
                                     h1 = model_GS_A2_3[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

    data_GS_A2_4.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_GS_A2_4[i,1], 
                                     sigma2 = model_GS_A2_4[i,2], 
                                     h1 = model_GS_A2_4[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

for i in range(npos):
    data_GS_Q_A2_1.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_A2_1[i,1], 
                                     sigma2 = model_GS_Q_A2_1[i,2], 
                                     h1 = model_GS_Q_A2_1[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_Q_A2_2.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_A2_2[i,1], 
                                     sigma2 = model_GS_Q_A2_2[i,2], 
                                     h1 = model_GS_Q_A2_2[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_Q_A2_3.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_A2_3[i,1], 
                                     sigma2 = model_GS_Q_A2_3[i,2], 
                                     h1 = model_GS_Q_A2_3[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

    data_GS_Q_A2_4.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_GS_Q_A2_4[i,1], 
                                     sigma2 = model_GS_Q_A2_4[i,2], 
                                     h1 = model_GS_Q_A2_4[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))
    
for i in range(npos):
    data_GS_IP_A2_1.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_A2_1[i,1], 
                                     sigma2 = model_GS_IP_A2_1[i,2], 
                                     h1 = model_GS_IP_A2_1[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_IP_A2_2.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_A2_2[i,1], 
                                     sigma2 = model_GS_IP_A2_2[i,2], 
                                     h1 = model_GS_IP_A2_2[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_GS_IP_A2_3.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_A2_3[i,1], 
                                     sigma2 = model_GS_IP_A2_3[i,2], 
                                     h1 = model_GS_IP_A2_3[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

    data_GS_IP_A2_4.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_GS_IP_A2_4[i,1], 
                                     sigma2 = model_GS_IP_A2_4[i,2], 
                                     h1 = model_GS_IP_A2_4[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))
    
# Simulate data in each position
for i in range(npos):
    data_Opt_A2_1.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_Opt_A2_1[i,1], 
                                     sigma2 = model_Opt_A2_1[i,2], 
                                     h1 = model_Opt_A2_1[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_Opt_A2_2.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_Opt_A2_2[i,1], 
                                     sigma2 = model_Opt_A2_2[i,2], 
                                     h1 = model_Opt_A2_2[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_Opt_A2_3.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_Opt_A2_3[i,1], 
                                     sigma2 = model_Opt_A2_3[i,2], 
                                     h1 = model_Opt_A2_3[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

    data_Opt_A2_4.append(EMf_2Lay_HVP(lambd, 
                                     sigma1 = model_Opt_A2_4[i,1], 
                                     sigma2 = model_Opt_A2_4[i,2], 
                                     h1 = model_Opt_A2_4[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

for i in range(npos):
    data_Opt_Q_A2_1.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_Opt_Q_A2_1[i,1], 
                                     sigma2 = model_Opt_Q_A2_1[i,2], 
                                     h1 = model_Opt_Q_A2_1[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_Opt_Q_A2_2.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_Opt_Q_A2_2[i,1], 
                                     sigma2 = model_Opt_Q_A2_2[i,2], 
                                     h1 = model_Opt_Q_A2_2[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_Opt_Q_A2_3.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_Opt_Q_A2_3[i,1], 
                                     sigma2 = model_Opt_Q_A2_3[i,2], 
                                     h1 = model_Opt_Q_A2_3[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

    data_Opt_Q_A2_4.append(EMf_2Lay_HVP_Q(lambd, 
                                     sigma1 = model_Opt_Q_A2_4[i,1], 
                                     sigma2 = model_Opt_Q_A2_4[i,2], 
                                     h1 = model_Opt_Q_A2_4[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))
    
for i in range(npos):
    data_Opt_IP_A2_1.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_Opt_IP_A2_1[i,1], 
                                     sigma2 = model_Opt_IP_A2_1[i,2], 
                                     h1 = model_Opt_IP_A2_1[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_Opt_IP_A2_2.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_Opt_IP_A2_2[i,1], 
                                     sigma2 = model_Opt_IP_A2_2[i,2], 
                                     h1 = model_Opt_IP_A2_2[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt)) 
    
    data_Opt_IP_A2_3.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_Opt_IP_A2_3[i,1], 
                                     sigma2 = model_Opt_IP_A2_3[i,2], 
                                     h1 = model_Opt_IP_A2_3[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))

    data_Opt_IP_A2_4.append(EMf_2Lay_HVP_IP(lambd, 
                                     sigma1 = model_Opt_IP_A2_4[i,1], 
                                     sigma2 = model_Opt_IP_A2_4[i,2], 
                                     h1 = model_Opt_IP_A2_4[i,0], 
                                     height = height,
                                     offsets = offsets,
                                     freq = freq,
                                     filt = filt))


# Store data and model
np.save('data/data_GS_A2_1', data_GS_A2_1)
np.save('data/data_GS_A2_2', data_GS_A2_2)
np.save('data/data_GS_A2_3', data_GS_A2_3)
np.save('data/data_GS_A2_4', data_GS_A2_4)

np.save('data/data_GS_Q_A2_1', data_GS_Q_A2_1)
np.save('data/data_GS_Q_A2_2', data_GS_Q_A2_2)
np.save('data/data_GS_Q_A2_3', data_GS_Q_A2_3)
np.save('data/data_GS_Q_A2_4', data_GS_Q_A2_4)

np.save('data/data_GS_IP_A2_1', data_GS_IP_A2_1)
np.save('data/data_GS_IP_A2_2', data_GS_IP_A2_2)
np.save('data/data_GS_IP_A2_3', data_GS_IP_A2_3)
np.save('data/data_GS_IP_A2_4', data_GS_IP_A2_4)

np.save('data/data_Opt_A2_1', data_Opt_A2_1)
np.save('data/data_Opt_A2_2', data_Opt_A2_2)
np.save('data/data_Opt_A2_3', data_Opt_A2_3)
np.save('data/data_Opt_A2_4', data_Opt_A2_4)

np.save('data/data_Opt_Q_A2_1', data_Opt_Q_A2_1)
np.save('data/data_Opt_Q_A2_2', data_Opt_Q_A2_2)
np.save('data/data_Opt_Q_A2_3', data_Opt_Q_A2_3)
np.save('data/data_Opt_Q_A2_4', data_Opt_Q_A2_4)

np.save('data/data_Opt_IP_A2_1', data_Opt_IP_A2_1)
np.save('data/data_Opt_IP_A2_2', data_Opt_IP_A2_2)
np.save('data/data_Opt_IP_A2_3', data_Opt_IP_A2_3)
np.save('data/data_Opt_IP_A2_4', data_Opt_IP_A2_4)