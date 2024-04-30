# Script to perform gradient based inversion for case A.1

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling classes for 2-layered models
from EM1D import EMf_2Lay_Opt_HVP_1D, EMf_2Lay_Opt_HVP_Q_1D, EMf_2Lay_Opt_HVP_IP_1D

# Import true models and data 
model_A1_1 = np.load('models/model_synth_2Lay_A1_1.npy')
model_A1_2 = np.load('models/model_synth_2Lay_A1_2.npy')
model_A1_3 = np.load('models/model_synth_2Lay_A1_3.npy')
model_A1_4 = np.load('models/model_synth_2Lay_A1_4.npy')

data_A1_1 = np.load('data/data_synth_2Lay_A1_1.npy')
data_A1_2 = np.load('data/data_synth_2Lay_A1_2.npy')
data_A1_3 = np.load('data/data_synth_2Lay_A1_3.npy')
data_A1_4 = np.load('data/data_synth_2Lay_A1_4.npy')

# Number of 1D models
npos = len(data_A1_1) # number of positions

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

m0 = [3, 500/1000, 500/1000]
lam = 0

## Case A1-1
# Optimization Q + IP

print('Estimating model A1-1 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_1[0]) * error
model_Opt_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_A1_1', model_Opt_A1_1)

# Optimization Q 

print('Estimating model A1-1 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_1[0, :9]) * error
model_Opt_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_A1_1', model_Opt_A1_1)

# Optimization IP

print('Estimating model A1-1 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_1[0, 9:]) * error
model_Opt_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_1[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_A1_1', model_Opt_A1_1)

## Case A1-2
# Optimization Q + IP

print('Estimating model A1-2 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_2[0]) * error
model_Opt_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_A1_2', model_Opt_A1_2)

# Optimization Q 

print('Estimating model A1-2 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_2[0, :9]) * error
model_Opt_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_A1_2', model_Opt_A1_2)

# Optimization IP

print('Estimating model A1-2 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_2[0, 9:]) * error
model_Opt_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_2[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_A1_2', model_Opt_A1_2)

## Case A1-3
# Optimization Q + IP

print('Estimating model A1-3 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_3[0]) * error
model_Opt_A1_3 = np.zeros_like(model_A1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_3[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_A1_3', model_Opt_A1_3)

# Optimization Q 

print('Estimating model A1-3 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_3[0, :9]) * error
model_Opt_A1_3 = np.zeros_like(model_A1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_3[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_A1_3', model_Opt_A1_3)

# Optimization IP

print('Estimating model A1-3 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_3[0, 9:]) * error
model_Opt_A1_3 = np.zeros_like(model_A1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_3[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_3[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_A1_3', model_Opt_A1_3)

## Case A1-4
# Optimization Q + IP

print('Estimating model A1-4 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_4[0]) * error
model_Opt_A1_4 = np.zeros_like(model_A1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_4[pos].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_A1_4', model_Opt_A1_4)

# Optimization Q 

print('Estimating model A1-4 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_4[0, :9]) * error
model_Opt_A1_4 = np.zeros_like(model_A1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_4[pos, :9].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_Q_A1_4', model_Opt_A1_4)

# Optimization IP

print('Estimating model A1-4 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_Opt_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

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
relativeError = np.ones_like(data_A1_4[0, 9:]) * error
model_Opt_A1_4 = np.zeros_like(model_A1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_4[pos, 9:].copy()
    model_Opt_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_Opt_A1_4[pos] = model_Opt_pos
print('End')
np.save('results/model_Opt_IP_A1_4', model_Opt_A1_4)