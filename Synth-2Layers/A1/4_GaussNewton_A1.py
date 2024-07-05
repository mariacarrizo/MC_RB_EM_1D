#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 4_GaussNewton_A1.py
Description: Script to perform Gauss-Newton inversion for A1 cases
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
model_A1_1 = np.load('models/model_synth_2Lay_A1_1.npy')
model_A1_2 = np.load('models/model_synth_2Lay_A1_2.npy')
model_A1_3 = np.load('models/model_synth_2Lay_A1_3.npy')
model_A1_4 = np.load('models/model_synth_2Lay_A1_4.npy')

data_A1_1 = np.load('data/data_synth_2Lay_A1_1.npy')
data_A1_2 = np.load('data/data_synth_2Lay_A1_2.npy')
data_A1_3 = np.load('data/data_synth_2Lay_A1_3.npy')
data_A1_4 = np.load('data/data_synth_2Lay_A1_4.npy')

# Number of 1D models positions
npos = len(data_A1_1) 

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

## Case A1-1
# Gauss-Newton inversion Q + IP

print('Estimating model A1-1 using Q+IP')
# Initialize the forward modelling class 
EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

# Inserting transformations in modelling class
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

print('Initializing inversion')
# Define inversion framework from pygimli
invEM = pg.Inversion()
invEM.setForwardOperator(EMf) # set forward operator

# Relative error array
relativeError = np.ones_like(data_A1_1[0]) * error

# Empty array to store estimated model A1-1
model_GN_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A1_1[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A1_1 = invEM.modelHistory # Store the optimization history of position 10
print('End')

# Q 

print('Estimating model A1-1 using Q')
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
relativeError = np.ones_like(data_A1_1[0, :9]) * error

# Empty array to store estimated model A1-1
model_GN_Q_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos, :9].copy() # Only Q data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A1_1[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A1_1 = invEM.modelHistory
print('End')

# IP

print('Estimating model A1-1 using IP')
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
relativeError = np.ones_like(data_A1_1[0, 9:]) * error

# Empty array to store estimated model A1-1
model_GN_IP_A1_1 = np.zeros_like(model_A1_1)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_1[pos, 9:].copy() # Only In-phase data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A1_1[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A1_1 = invEM.modelHistory
print('End')

## Case A1-2
# Gauss-Newton inversion Q + IP

print('Estimating model A1-2 using Q+IP')
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
relativeError = np.ones_like(data_A1_2[0]) * error

# Empty array to store estimated model A1-2
model_GN_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A1_2[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A1_2 = invEM.modelHistory
print('End')

# Q 

print('Estimating model A1-2 using Q')
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
relativeError = np.ones_like(data_A1_2[0, :9]) * error

# Empty array to store estimated model A1-2
model_GN_Q_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos, :9].copy() # Only Quadrature data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A1_2[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A1_2 = invEM.modelHistory
print('End')

# IP

print('Estimating model A1-2 using IP')
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
relativeError = np.ones_like(data_A1_2[0, 9:]) * error

# Empty array to store estimated model A1-2
model_GN_IP_A1_2 = np.zeros_like(model_A1_2)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_2[pos, 9:].copy() # Only In-Phase data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A1_2[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A1_2 = invEM.modelHistory
print('End')

## Case A1-3
# Gauss-Newton inversion Q + IP

print('Estimating model A1-3 using Q+IP')
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
relativeError = np.ones_like(data_A1_3[0]) * error

# Empty array to store estimated model A1-3
model_GN_A1_3 = np.zeros_like(model_A1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_3[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A1_3[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A1_3 = invEM.modelHistory
print('End')

# Q 

print('Estimating model A1-3 using Q')
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
relativeError = np.ones_like(data_A1_3[0, :9]) * error

# Empty array to store estimated model A1-3
model_GN_Q_A1_3 = np.zeros_like(model_A1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_3[pos, :9].copy() # Only quadrature data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A1_3[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A1_3 = invEM.modelHistory
print('End')

# IP

print('Estimating model A1-3 using IP')
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
relativeError = np.ones_like(data_A1_3[0, 9:]) * error

# Empty array to store estimated model A1-3
model_GN_IP_A1_3 = np.zeros_like(model_A1_3)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_3[pos, 9:].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A1_3[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A1_3 = invEM.modelHistory
print('End')

## Case A1-4
# Gauss-Newton inversion using Q + IP

print('Estimating model A1-4 using Q+IP')
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
relativeError = np.ones_like(data_A1_4[0]) * error

# Empty array to store estimated model A1-4
model_GN_A1_4 = np.zeros_like(model_A1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_4[pos].copy()
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_A1_4[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_A1_4 = invEM.modelHistory
print('End')

# Q 

print('Estimating model A1-4 using Q')
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
relativeError = np.ones_like(data_A1_4[0, :9]) * error

# Empty array to store estimated model A1-4
model_GN_Q_A1_4 = np.zeros_like(model_A1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_4[pos, :9].copy() # only quadrature data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_Q_A1_4[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_Q_A1_4 = invEM.modelHistory
print('End')

# IP

print('Estimating model A1-4 using IP')
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
relativeError = np.ones_like(data_A1_4[0, 9:]) * error

# Empty array to store estimated model A1-4
model_GN_IP_A1_4 = np.zeros_like(model_A1_4)

# Start inversion
print('Run inversion')
# Perform inversion for each 1D model 
for pos in range(npos):
    dataE = data_A1_4[pos, 9:].copy() # only in-phase data
    model_GN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)
    model_GN_IP_A1_4[pos] = model_GN_pos
    if pos == pos_test:
        mod_hist_IP_A1_4 = invEM.modelHistory
print('End')

# Store A1-1 estimations
np.save('results/model_GN_A1_1', model_GN_A1_1)
np.save('results/model_GN_A1_1_hist', mod_hist_A1_1)
np.save('results/model_GN_Q_A1_1', model_GN_Q_A1_1)
np.save('results/model_GN_A1_1_hist_Q', mod_hist_Q_A1_1)
np.save('results/model_GN_IP_A1_1', model_GN_IP_A1_1)
np.save('results/model_GN_A1_1_hist_IP', mod_hist_IP_A1_1)

# Store A1-2 estimations
np.save('results/model_GN_A1_2', model_GN_A1_2)
np.save('results/model_GN_A1_2_hist', mod_hist_A1_2)
np.save('results/model_GN_Q_A1_2', model_GN_Q_A1_2)
np.save('results/model_GN_A1_2_hist_Q', mod_hist_Q_A1_2)
np.save('results/model_GN_IP_A1_2', model_GN_IP_A1_2)
np.save('results/model_GN_A1_2_hist_IP', mod_hist_IP_A1_2)

# Store A1-3 estimations
np.save('results/model_GN_A1_3', model_GN_A1_3)
np.save('results/model_GN_A1_3_hist', mod_hist_A1_3)
np.save('results/model_GN_Q_A1_3', model_GN_Q_A1_3)
np.save('results/model_GN_A1_3_hist_Q', mod_hist_Q_A1_3)
np.save('results/model_GN_IP_A1_3', model_GN_IP_A1_3)
np.save('results/model_GN_A1_3_hist_IP', mod_hist_IP_A1_3)

# Store A1-4 estimations
np.save('results/model_GN_A1_4', model_GN_A1_4)
np.save('results/model_GN_A1_4_hist', mod_hist_A1_4)
np.save('results/model_GN_Q_A1_4', model_GN_Q_A1_4)
np.save('results/model_GN_A1_4_hist_Q', mod_hist_Q_A1_4)
np.save('results/model_GN_IP_A1_4', model_GN_IP_A1_4)
np.save('results/model_GN_A1_4_hist_IP', mod_hist_IP_A1_4)