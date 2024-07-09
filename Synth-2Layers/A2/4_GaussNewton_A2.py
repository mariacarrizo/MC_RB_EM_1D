#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 4_GaussNewton_A2.py
Description: Script to perform Gauss-Newton inversion for A2 cases
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

# import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling classes for 2-layered models
from EM1D import EMf_2Lay_GN_HVP_1D, EMf_2Lay_GN_HVP_Q_1D, EMf_2Lay_GN_HVP_IP_1D

# Import true models and data 
model_A2_1 = np.load('models/model_synth_2Lay_A2_1.npy')
model_A2_2 = np.load('models/model_synth_2Lay_A2_2.npy')
model_A2_3 = np.load('models/model_synth_2Lay_A2_3.npy')
model_A2_4 = np.load('models/model_synth_2Lay_A2_4.npy')

data_A2_1 = np.load('data/data_synth_2Lay_A2_1.npy')
data_A2_2 = np.load('data/data_synth_2Lay_A2_2.npy')
data_A2_3 = np.load('data/data_synth_2Lay_A2_3.npy')
data_A2_4 = np.load('data/data_synth_2Lay_A2_4.npy')

# Number of 1D models
npos = len(data_A2_1) # number of positions

# Load survey parameters
survey = np.load('../data/survey_2Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Define initial model [h_1, sigma_1, sigma_2] (sigmas in S/m)
m0 = [3, 500/1000, 500/1000]

# Define regularization parameter (alpha in Equation 5)
lam = 0

# relative error for inversion class
error = 1e-3 

# Define a position that you want to check the optimization history
pos_test = 10

# Defining inversion limits and transformations
transThk = pg.trans.TransLogLU(0.1,7)
transSig = pg.trans.TransLogLU(10/1000,2000/1000)

## Case A2-1
# Gauss-Newton Q + IP

print('Estimating model A2-1 using Q+IP')
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
relativeError = np.ones_like(data_A2_1[0]) * error
model_GN_A2_1 = np.zeros_like(model_A2_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A2_1[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A2_1 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model A2-1 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_1[0, :9]) * error
model_GN_Q_A2_1 = np.zeros_like(model_A2_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_1[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A2_1[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A2_1 = invEM.modelHistory
print('End')

# Gauss-Newton IP

print('Estimating model A2-1 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_1[0, 9:]) * error
model_GN_IP_A2_1 = np.zeros_like(model_A2_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_1[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A2_1[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A2_1 = invEM.modelHistory
print('End')

## Case A2-2
# Gauss-Newton Q + IP

print('Estimating model A2-2 using Q+IP')
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
relativeError = np.ones_like(data_A2_2[0]) * error
model_GN_A2_2 = np.zeros_like(model_A2_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A2_2[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A2_2 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model A2-2 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_2[0, :9]) * error
model_GN_Q_A2_2 = np.zeros_like(model_A2_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_2[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A2_2[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A2_2 = invEM.modelHistory
print('End')

# Gauss-Newton IP

print('Estimating model A2-2 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_2[0, 9:]) * error
model_GN_IP_A2_2 = np.zeros_like(model_A2_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_2[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A2_2[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A2_2 = invEM.modelHistory
print('End')

## Case A2-3
# Gauss-Newton Q + IP

print('Estimating model A2-3 using Q+IP')
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
relativeError = np.ones_like(data_A2_3[0]) * error
model_GN_A2_3 = np.zeros_like(model_A2_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A2_3[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A2_3 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model A2-3 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_3[0, :9]) * error
model_GN_Q_A2_3 = np.zeros_like(model_A2_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_3[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A2_3[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A2_3 = invEM.modelHistory
print('End')

# Gauss-Newton IP

print('Estimating model A2-3 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_3[0, 9:]) * error
model_GN_IP_A2_3 = np.zeros_like(model_A2_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_3[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A2_3[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A2_3 = invEM.modelHistory
print('End')

## Case A2-4
# Gauss-Newton Q + IP

print('Estimating model A2-4 using Q+IP')
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
relativeError = np.ones_like(data_A2_4[0]) * error
model_GN_A2_4 = np.zeros_like(model_A2_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A2_4[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A2_4 = invEM.modelHistory
print('End')

# Gauss-Newton Q 

print('Estimating model A2-4 using Q')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_Q_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_4[0, :9]) * error
model_GN_Q_A2_4 = np.zeros_like(model_A2_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_4[pos, :9].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A2_4[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A2_4 = invEM.modelHistory
print('End')

# Gauss-Newton IP

print('Estimating model A2-4 using IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_IP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A2_4[0, 9:]) * error
model_GN_IP_A2_4 = np.zeros_like(model_A2_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A2_4[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A2_4[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A2_4 = invEM.modelHistory
print('End')

# Store the estimated models
np.save('results/model_GN_A2_1', model_GN_A2_1)
np.save('results/model_GN_A2_1_hist', mod_hist_A2_1)
np.save('results/model_GN_Q_A2_1', model_GN_Q_A2_1)
np.save('results/model_GN_A2_1_hist_Q', mod_hist_Q_A2_1)
np.save('results/model_GN_IP_A2_1', model_GN_IP_A2_1)
np.save('results/model_GN_A2_1_hist_IP', mod_hist_IP_A2_1)

np.save('results/model_GN_A2_2', model_GN_A2_2)
np.save('results/model_GN_A2_2_hist', mod_hist_A2_2)
np.save('results/model_GN_Q_A2_2', model_GN_Q_A2_2)
np.save('results/model_GN_A2_2_hist_Q', mod_hist_Q_A2_2)
np.save('results/model_GN_IP_A2_2', model_GN_IP_A2_2)
np.save('results/model_GN_A2_2_hist_IP', mod_hist_IP_A2_2)

np.save('results/model_GN_A2_3', model_GN_A2_3)
np.save('results/model_GN_A2_3_hist', mod_hist_A2_3)
np.save('results/model_GN_Q_A2_3', model_GN_Q_A2_3)
np.save('results/model_GN_A2_3_hist_Q', mod_hist_Q_A2_3)
np.save('results/model_GN_IP_A2_3', model_GN_IP_A2_3)
np.save('results/model_GN_A2_3_hist_IP', mod_hist_IP_A2_3)

np.save('results/model_GN_A2_4', model_GN_A2_4)
np.save('results/model_GN_A2_4_hist', mod_hist_A2_4)
np.save('results/model_GN_Q_A2_4', model_GN_Q_A2_4)
np.save('results/model_GN_A2_4_hist_Q', mod_hist_Q_A2_4)
np.save('results/model_GN_IP_A2_4', model_GN_IP_A2_4)
np.save('results/model_GN_A2_4_hist_IP', mod_hist_IP_A2_4)