# Testing optimization case A.1 with noisy data 

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling class for 2-layered models
from EM1D import EMf_3Lay_Opt_HVP_1D

# Import the conductivities and thicknesses used to create the LU table
#conds = np.load('../data/conds.npy')
#thick = np.load('../data/thicks.npy')

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Import true models and data 
model_B1_1 = np.load('models/model_synth_B1_1.npy')
model_B1_2 = np.load('models/model_synth_B1_2.npy')
model_B1_3 = np.load('models/model_synth_B1_3.npy')
model_B1_4 = np.load('models/model_synth_B1_4.npy')

data_B1_1 = np.load('data/data_synth_B1_1.npy')
data_B1_2 = np.load('data/data_synth_B1_2.npy')
data_B1_3 = np.load('data/data_synth_B1_3.npy')
data_B1_4 = np.load('data/data_synth_B1_4.npy')

# number of 1D models 
npos = len(data) 

# Load data with added noise
data_n2_B1_1 = np.load('data/data_n2_B1_1.npy')
data_n2_B1_2 = np.load('data/data_n2_B1_2.npy')
data_n2_B1_3 = np.load('data/data_n2_B1_3.npy')
data_n2_B1_4 = np.load('data/data_n2_B1_4.npy')

data_n5_B1_1 = np.load('data/data_n5_B1_1.npy')
data_n5_B1_2 = np.load('data/data_n5_B1_2.npy')
data_n5_B1_3 = np.load('data/data_n5_B1_3.npy')
data_n5_B1_4 = np.load('data/data_n5_B1_4.npy')

data_n10_B1_1 = np.load('data/data_n10_B1_1.npy')
data_n10_B1_2 = np.load('data/data_n10_B1_2.npy')
data_n10_B1_3 = np.load('data/data_n10_B1_3.npy')
data_n10_B1_4 = np.load('data/data_n10_B1_4.npy')

#%%
# Optimization for data [Q + IP] noise 2.5 %
# Initialize the forward modelling class
m0 = [3, 3, 500/1000, 500/1000, 500/1000]
lam = 0

## Case A1-1
print('Estimating model n2 B1-1')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n2_B1_1[0]) * error
model_Opt_n2_B1_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_B1_1[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n2_B1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n2_B1_1', model_Opt_n2_B1_1)

print('Estimating model n2 B1-2')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n2_B1_2[0]) * error
model_Opt_n2_B1_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_B1_2[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n2_B1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n2_B1_2', model_Opt_n2_B1_2)

print('Estimating model n2 B1-3')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n2_B1_3[0]) * error
model_Opt_n2_B1_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_B1_3[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n2_B1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n2_B1_3', model_Opt_n2_B1_3)

print('Estimating model n2 B1-4')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n2_B1_4[0]) * error
model_Opt_n2_B1_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_B1_4[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n2_B1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n2_B1_4', model_Opt_n2_B1_4)

## Optimization for data [Q + IP] noise 5 %
print('Estimating model n5 B1-1')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n5_B1_1[0]) * error
model_Opt_n5_B1_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_B1_1[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n5_B1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n5_B1_1', model_Opt_n5_B1_1)

print('Estimating model n5 B1-2')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n5_B1_2[0]) * error
model_Opt_n5_B1_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_B1_2[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n5_B1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n5_B1_2', model_Opt_n5_B1_2)

print('Estimating model n5 B1-3')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n5_B1_3[0]) * error
model_Opt_n5_B1_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_B1_3[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n5_B1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n5_B1_3', model_Opt_n5_B1_3)

print('Estimating model n5 B1-4')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n5_B1_4[0]) * error
model_Opt_n5_B1_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_B1_4[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n5_B1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n5_B1_4', model_Opt_n5_B1_4)

## Optimization for data [Q + IP] noise 5 %
print('Estimating model n10 B1-1')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n10_B1_1[0]) * error
model_Opt_n10_B1_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_B1_1[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n10_B1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n10_B1_1', model_Opt_n10_B1_1)

print('Estimating model n10 B1-2')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n10_B1_2[0]) * error
model_Opt_n10_B1_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_B1_2[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n10_B1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n10_B1_2', model_Opt_n10_B1_2)

print('Estimating model n10 B1-3')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n10_B1_3[0]) * error
model_Opt_n10_B1_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_B1_3[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n10_B1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n10_B1_3', model_Opt_n10_B1_3)

print('Estimating model n10 B1-4')
# Initialize the forward modelling class 
EMf = EMf_3Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

transThk = pg.trans.TransLogLU(0.1,7)
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
relativeError = np.ones_like(data_n10_B1_4[0]) * error
model_Opt_n10_B1_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_B1_4[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_n10_B1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_n10_B1_4', model_Opt_n10_B1_4)
