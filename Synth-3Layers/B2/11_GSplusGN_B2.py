#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 11_GSplusGN_B2.py
Description: Script to perform Global Search + Gauss-Newton inversion for B2 cases 
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""

# Import libraries
import pygimli as pg
import numpy as np
import sys
sys.path.insert(1, '../../src')

# Import forward modelling class for 3-layered 1D models for GS + GN algorithm
from EM1D import EMf_3Lay_GN_HVP_1D

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('../data/conds.npy')
thick = np.load('../data/thicks.npy')

# Load survey parameters
survey = np.load('../data/survey_3Lay.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Data array for all the 1D stitched models
data_B2_1 = np.load('data/data_synth_B2_1.npy')
data_B2_2 = np.load('data/data_synth_B2_2.npy')
data_B2_3 = np.load('data/data_synth_B2_3.npy')
data_B2_4 = np.load('data/data_synth_B2_4.npy')

npos = len(data_B2_1) # number of 1D models

# Import model from Global search
model_GS_B2_1 = np.load('results/model_GS_B2_1.npy')
model_GS_B2_2 = np.load('results/model_GS_B2_2.npy')
model_GS_B2_3 = np.load('results/model_GS_B2_3.npy')
model_GS_B2_4 = np.load('results/model_GS_B2_4.npy')

transThk = pg.trans.TransLogLU(1, 4)
transSig = pg.trans.TransLogLU(10/1000, 2000/1000)

# Define regularization parameter (alpha in Equation 5)
lam=0

# relative error for inversion class
error = 1e-3 

# Case B2-1

# Relative error array
relativeError = np.ones_like(data_B2_1[0]) * error
model_GSGN_B2_1 = np.zeros_like(model_GS_B2_1)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B2_1[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B2_1[pos].copy()
    model_GSGN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GSGN_B2_1[pos] = model_GSGN_pos

# Case B2-2

# Relative error array
relativeError = np.ones_like(data_B2_2[0]) * error
model_GSGN_B2_2 = np.zeros_like(model_GS_B2_2)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B2_2[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B2_2[pos].copy()
    model_GSGN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GSGN_B2_2[pos] = model_GSGN_pos

# Case B2-3

# Relative error array
relativeError = np.ones_like(data_B2_3[0]) * error
model_GSGN_B2_3 = np.zeros_like(model_GS_B2_3)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B2_3[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B2_3[pos].copy()
    model_GSGN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GSGN_B2_3[pos] = model_GSGN_pos

# Case B2-4   

# Relative error array
relativeError = np.ones_like(data_B2_4[0]) * error
model_GSGN_B2_4 = np.zeros_like(model_GS_B2_4)

# Start inversion
# Perform inversion for each 1D model per position in stitched section
for pos in range(npos):
    # Set the initial model from the global search
    m0 = model_GS_B2_4[pos]
    
    # Initialize the forward modelling class
    EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)
    
    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    dataE = data_B2_4[pos].copy()
    model_GSGN_pos = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=True)
    model_GSGN_B2_4[pos] = model_GSGN_pos

# Save estimated models    
np.save('results/model_GSGN_B2_1', model_GSGN_B2_1)
np.save('results/model_GSGN_B2_2', model_GSGN_B2_2)
np.save('results/model_GSGN_B2_3', model_GSGN_B2_3)
np.save('results/model_GSGN_B2_4', model_GSGN_B2_4)