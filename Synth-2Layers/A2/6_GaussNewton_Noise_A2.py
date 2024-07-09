#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 6_GaussNewton_Noise_A2.py
Description: Script to test Gauss-Newton inversion in A2 cases with noisy data 
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 18/12/2023
"""

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling class for 2-layered models
from EM1D import EMf_2Lay_GN_HVP_1D

# Import the conductivities and thicknesses used to create the LU table
#conds = np.load('../data/conds.npy')
#thick = np.load('../data/thicks.npy')

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Import true models and data 
model = np.load('models/model_synth_2Lay_A2_1.npy')
data = np.load('data/data_synth_2Lay_A2_1.npy')

# number of 1D models positions
npos = len(data) 

# Load data with added noise
data_n2_A2_1 = np.load('data/data_n2_A2_1.npy')
data_n2_A2_2 = np.load('data/data_n2_A2_2.npy')
data_n2_A2_3 = np.load('data/data_n2_A2_3.npy')
data_n2_A2_4 = np.load('data/data_n2_A2_4.npy')

data_n5_A2_1 = np.load('data/data_n5_A2_1.npy')
data_n5_A2_2 = np.load('data/data_n5_A2_2.npy')
data_n5_A2_3 = np.load('data/data_n5_A2_3.npy')
data_n5_A2_4 = np.load('data/data_n5_A2_4.npy')

data_n10_A2_1 = np.load('data/data_n10_A2_1.npy')
data_n10_A2_2 = np.load('data/data_n10_A2_2.npy')
data_n10_A2_3 = np.load('data/data_n10_A2_3.npy')
data_n10_A2_4 = np.load('data/data_n10_A2_4.npy')

# Define initial model [h_1, sigma_1, sigma_2] (sigmas in S/m)
m0 = [3, 500/1000, 500/1000]

# Define regularization parameter (alpha in Equation 5)
lam = 0

# relative error for inversion class
error = 1e-3 

# Defining inversion limits and transformations
transThk = pg.trans.TransLogLU(0.1,7)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

# Gauss-Newton for data [Q + IP] noise 2.5 %

## Case A2-1
print('Estimating model n2 A2-1')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A2_1[0]) * error
model_GN_n2_A2_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A2_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A2_1[pos] = model_GN_pos
print('End')

# Case A2-2
print('Estimating model n2 A2-2')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A2_2[0]) * error
model_GN_n2_A2_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A2_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A2_2[pos] = model_GN_pos
print('End')

# Case A2-3
print('Estimating model n2 A2-3')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A2_3[0]) * error
model_GN_n2_A2_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A2_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A2_3[pos] = model_GN_pos
print('End')

# Case A2-4
print('Estimating model n2 A2-4')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n2_A2_4[0]) * error
model_GN_n2_A2_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n2_A2_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n2_A2_4[pos] = model_GN_pos
print('End')

## Gauss-Newton for data [Q + IP] noise 5 %

# Case A2-1
print('Estimating model n5 A2-1')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A2_1[0]) * error
model_GN_n5_A2_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A2_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A2_1[pos] = model_GN_pos
print('End')

# Case A2-2
print('Estimating model n5 A2-2')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A2_2[0]) * error
model_GN_n5_A2_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A2_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A2_2[pos] = model_GN_pos
print('End')

# Case A2-3
print('Estimating model n5 A2-3')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A2_3[0]) * error
model_GN_n5_A2_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A2_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A2_3[pos] = model_GN_pos
print('End')

# Case A2-4
print('Estimating model n5 A2-4')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n5_A2_4[0]) * error
model_GN_n5_A2_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n5_A2_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n5_A2_4[pos] = model_GN_pos
print('End')

## Gauss-Newton for data [Q + IP] noise 10 %

# Case A2-1
print('Estimating model n10 A2-1')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A2_1[0]) * error
model_GN_n10_A2_1 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A2_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A2_1[pos] = model_GN_pos
print('End')

# Case A2-2
print('Estimating model n10 A2-2')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A2_2[0]) * error
model_GN_n10_A2_2 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A2_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A2_2[pos] = model_GN_pos
print('End')

# Case A2-3
print('Estimating model n10 A2-3')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A2_3[0]) * error
model_GN_n10_A2_3 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A2_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A2_3[pos] = model_GN_pos
print('End')

# Case A2-4
print('Estimating model n10 A2-4')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_n10_A2_4[0]) * error
model_GN_n10_A2_4 = np.zeros_like(model)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_n10_A2_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_n10_A2_4[pos] = model_GN_pos
print('End')

# Store results
np.save('results/model_n2_GN_A2_1', model_GN_n2_A2_1)
np.save('results/model_n2_GN_A2_2', model_GN_n2_A2_2)
np.save('results/model_n2_GN_A2_3', model_GN_n2_A2_3)
np.save('results/model_n2_GN_A2_4', model_GN_n2_A2_4)

np.save('results/model_n5_GN_A2_1', model_GN_n5_A2_1)
np.save('results/model_n5_GN_A2_2', model_GN_n5_A2_2)
np.save('results/model_n5_GN_A2_3', model_GN_n5_A2_3)
np.save('results/model_n5_GN_A2_4', model_GN_n5_A2_4)

np.save('results/model_n10_GN_A2_1', model_GN_n10_A2_1)
np.save('results/model_n10_GN_A2_2', model_GN_n10_A2_2)
np.save('results/model_n10_GN_A2_3', model_GN_n10_A2_3)
np.save('results/model_n10_GN_A2_4', model_GN_n10_A2_4)