#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 6_GaussNewton_B1.py
Description: Script to perform Gauss-Newton inversion for B1 cases
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Load forward modelling classes for 3-layered 1D models
from EM1D import EMf_3Lay_GN_HVP_1D, EMf_3Lay_GN_HVP_Q_1D, EMf_3Lay_GN_HVP_IP_1D

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

# Define initial model [h_1, h2, sigma_1, sigma_2, sigma_3] (sigmas in S/m)
m0 = [3, 3, 100/1000, 100/1000, 100/1000]

# Define regularization parameter (alpha in Equation 5)
lam = 0

# relative error for inversion class
error = 1e-3 

# Define a position that you want to check the optimization history
pos_test = 10

# Defining inversion limits and transformations
transThk = pg.trans.TransLogLU(1, 4)
transSig = pg.trans.TransLogLU(10/1000, 2000/1000)

#%%
# Gauss-Newton Q + IP

print('Estimating model B1-1 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_1[0]) * error
model_GN_B1_1 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    print('pos:', pos)
    dataE = data_B1_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    if pos == pos_test:
        model_hist_B1_1 = invEM.modelHistory
    model_GN_B1_1[pos] = model_GN_pos
print('End')

# Gauss-Newton Q 

print('Estimating model B1-1 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_1[0, :9]) * error
model_GN_Q_B1_1 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_1[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_B1_1[pos] = model_GN_pos
print('End')

# Gauss-Newton IP 

print('Estimating model B1-1 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_1[0, 9:]) * error
model_GN_IP_B1_1 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_1[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_B1_1[pos] = model_GN_pos
print('End')

# Gauss-Newton Q + IP

print('Estimating model B1-2 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_2[0]) * error
model_GN_B1_2 = np.zeros_like(model_B1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    print('pos:', pos)
    dataE = data_B1_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GN_B1_2[pos] = model_GN_pos
    if pos == pos_test:
        model_hist_B1_2 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model B1-2 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_2[0, :9]) * error
model_GN_Q_B1_2 = np.zeros_like(model_B1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_2[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_B1_2[pos] = model_GN_pos
print('End')

# Gauss-Newton IP 

print('Estimating model B1-2 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_2[0, 9:]) * error
model_GN_IP_B1_2 = np.zeros_like(model_B1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_2[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_B1_2[pos] = model_GN_pos
print('End')

# Gauss-Newton Q + IP

print('Estimating model B1-3 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_3[0]) * error
model_GN_B1_3 = np.zeros_like(model_B1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    print('pos:', pos)
    dataE = data_B1_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GN_B1_3[pos] = model_GN_pos
    if pos == pos_test:
        model_hist_B1_3 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model B1-3 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_3[0, :9]) * error
model_GN_Q_B1_3 = np.zeros_like(model_B1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_3[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_B1_3[pos] = model_GN_pos
print('End')

# Gauss-Newton IP 

print('Estimating model B1-3 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_3[0, 9:]) * error
model_GN_IP_B1_3 = np.zeros_like(model_B1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_3[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_B1_3[pos] = model_GN_pos
print('End')

# Gauss-Newton Q + IP

print('Estimating model B1-4 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_4[0]) * error
model_GN_B1_4 = np.zeros_like(model_B1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    print('pos:', pos)
    dataE = data_B1_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GN_B1_4[pos] = model_GN_pos
    if pos == pos_test:
        model_hist_B1_4 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model B1-4 using Q')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_4[0, :9]) * error
model_GN_Q_B1_4 = np.zeros_like(model_B1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_4[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_B1_4[pos] = model_GN_pos
print('End')

# Gauss-Newton IP 

print('Estimating model B1-4 using IP')
# Initialize the forward modelling class 
EMf = EMf_3Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=3)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_B1_4[0, 9:]) * error
model_GN_IP_B1_4 = np.zeros_like(model_B1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_B1_4[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_B1_4[pos] = model_GN_pos
print('End')

# Store estimations
np.save('results/model_GN_B1_1', model_GN_B1_1)
np.save('results/model_GN_B1_1_hist', model_hist_B1_1)
np.save('results/model_GN_Q_B1_1', model_GN_Q_B1_1)
np.save('results/model_GN_IP_B1_1', model_GN_IP_B1_1)

np.save('results/model_GN_B1_2', model_GN_B1_2)
np.save('results/model_GN_B1_2_hist', model_hist_B1_2)
np.save('results/model_GN_Q_B1_2', model_GN_Q_B1_2)
np.save('results/model_GN_IP_B1_2', model_GN_IP_B1_2)

np.save('results/model_GN_B1_3', model_GN_B1_3)
np.save('results/model_GN_B1_3_hist', model_hist_B1_3)
np.save('results/model_GN_Q_B1_3', model_GN_Q_B1_3)
np.save('results/model_GN_IP_B1_3', model_GN_IP_B1_3)

np.save('results/model_GN_B1_4', model_GN_B1_4)
np.save('results/model_GN_B1_4_hist', model_hist_B1_4)
np.save('results/model_GN_Q_B1_4', model_GN_Q_B1_4)
np.save('results/model_GN_IP_B1_4', model_GN_IP_B1_4)