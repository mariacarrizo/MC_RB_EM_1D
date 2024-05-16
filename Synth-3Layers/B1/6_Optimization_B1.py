# Script to perform gradient based inversion on 3-layered 1D models

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Load forward modelling classes for 3-layered 1D models
from EM1D import EMf_3Lay_Opt_HVP_1D, EMf_3Lay_Opt_HVP_Q_1D, EMf_3Lay_Opt_HVP_IP_1D

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Import true models and data 
model_B1_1 = np.load('models/model_synth_B1_1.npy')
model_B1_2 = np.load('models/model_synth_B1_2.npy')
model_B1_3 = np.load('models/model_synth_B1_3.npy')
model_B1_4 = np.load('models/model_synth_B1_4.npy')

# Data array for all the 1D stitched models
data_B1_1 = np.load('data/data_synth_B1_1.npy')
data_B1_2 = np.load('data/data_synth_B1_2.npy')
data_B1_3 = np.load('data/data_synth_B1_3.npy')
data_B1_4 = np.load('data/data_synth_B1_4.npy')

npos = len(data_B1_1) # number of 1D models

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

m0 = [3, 3, 500/1000, 500/1000, 500/1000]
lam = 0

#%%
# Optimization Q + IP

print('Estimating model B1-1 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_1[0]) * error
model_Opt_B1_1 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_1[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_B1_1', model_Opt_B1_1)

#%%
# Optimization Q 

print('Estimating model B1-1 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_1[0, :9]) * error
model_Opt_B1_1 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_1[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_B1_1', model_Opt_B1_1)
    
#%%    
# Optimization IP 

print('Estimating model B1-1 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_1[0, 9:]) * error
model_Opt_B1_1 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_1[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_B1_1', model_Opt_B1_1)

#%%
# Optimization Q + IP

print('Estimating model B1-2 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_2[0]) * error
model_Opt_B1_2 = np.zeros_like(model_B1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_2[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_B1_2', model_Opt_B1_2)

#%%
# Optimization Q 

print('Estimating model B1-2 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_2[0, :9]) * error
model_Opt_B1_2 = np.zeros_like(model_B1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_2[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_B1_2', model_Opt_B1_2)
    
#%%    
# Optimization IP 

print('Estimating model B1-2 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_2[0, 9:]) * error
model_Opt_B1_2 = np.zeros_like(model_B1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_2[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_B1_2', model_Opt_B1_2)

#%%
# Optimization Q + IP

print('Estimating model B1-3 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_3[0]) * error
model_Opt_B1_3 = np.zeros_like(model_B1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_3[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_B1_3', model_Opt_B1_3)

#%%
# Optimization Q 

print('Estimating model B1-3 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_3[0, :9]) * error
model_Opt_B1_3 = np.zeros_like(model_B1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_3[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_B1_3', model_Opt_B1_3)
    
#%%    
# Optimization IP 

print('Estimating model B1-3 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_3[0, 9:]) * error
model_Opt_B1_3 = np.zeros_like(model_B1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_3[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_B1_3', model_Opt_B1_3)

#%%
# Optimization Q + IP

print('Estimating model B1-4 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_4[0]) * error
model_Opt_B1_4 = np.zeros_like(model_B1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_4[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_B1_4', model_Opt_B1_4)

#%%
# Optimization Q 

print('Estimating model B1-4 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_4[0, :9]) * error
model_Opt_B1_4 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_4[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_B1_4', model_Opt_B1_4)
    
#%%    
# Optimization IP 

print('Estimating model B1-4 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(1.5,5.5)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
error = 1e-3 # relative error
relativeError = np.ones_like(data_B1_4[0, 9:]) * error
model_Opt_B1_4 = np.zeros_like(model_B1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_4[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_B1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_B1_4', model_Opt_B1_4)