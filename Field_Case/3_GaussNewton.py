#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: 3_GaussNewton.py
Description: Script to perform Gauss-Newton inversion for Field case
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 17/12/2023
"""


# Import libraries
import pygimli as pg
import numpy as np
import pandas as pd
import sys
sys.path.insert(1, '../src')

# Load forward modelling class for 2-layered models in field case
from EM1D import EMf_2Lay_GN_HV_field

# Import the conductivities and thicknesses used to create the LU table
conds = np.load('data/conds.npy')
thick = np.load('data/thicks.npy')

# Import field data
Dataframe = pd.DataFrame(np.load('data/Field_data.npy'),
                        columns = ['X','Y','Position','Z','H2Q','H4Q','H8Q',
                                   'V2Q','V4Q','V8Q','P2Q','P4Q','P8Q',
                                   'H4IP','H8IP','V4IP','V8IP'])

# Obtain H and V quadrature and in-phase measurements
# For in-phase we only use measurements for offsets > 4 m
data = np.array(pd.concat([Dataframe.loc[:,'H2Q':'V8Q'], Dataframe.loc[:,'H4IP':]], axis=1))
npos = len(data) # number of 1D models
nlay = 2 # number of layers

# Load survey parameters
survey = np.load('data/survey_field.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

# Define initial model [h_1, sigma_1, sigma_2] (sigmas in S/m)
m0 = [3, 100/1000, 100/1000]

# Define regularization parameter (alpha in Equation 5)
lam = 0

# relative error for inversion class
error = 1e-3 

# Defining inversion limits and transformations
transThk = pg.trans.TransLogLU(np.min(thick), np.max(thick))
transSig = pg.trans.TransLogLU(np.min(conds), np.max(conds))

# Optimization Q + IP

# Initialize the forward modelling class
EMf = EMf_2Lay_GN_HV_field(lambd, height, offsets, freq, filt, nlay=2)

# Define transformation
EMf.region(0).setTransModel(transThk)
EMf.region(1).setTransModel(transSig)

# Create inversion
invEM = pg.Inversion()
invEM.setForwardOperator(EMf)

# Relative error array

relativeError = np.ones_like(data[0]) * error
model_GN = np.zeros((npos, nlay+1))

# Start inversion
# Perform inversion for each 1D model per position 
for pos in range(npos):
    dataE = data[pos].copy()
    model_GN[pos] = invEM.run(dataE, relativeError, startModel= m0, lam=lam, verbose=False)

# Save estimated model
np.save('results/model_2Lay_GN_field', model_GN)