#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Script Name: Plots.py
Description: Script with model and data visualization functions 
Author: @mariacarrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 18/12/2023
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

import pygimli as pg
from pygimli.utils import rndig
from pygimli.viewer.mpl import setMappableData
from pygimli.viewer.mpl import updateAxes as updateAxes_

def showStitchedModels(models, ax=None, x=None, cMin=None, cMax=None, thk=None,
                       logScale=True, title=None, zMin=0, zMax=0, zLog=False,
                       **kwargs):
    """Show several 1d block models as (stitched) section.

    Parameters
    ----------
    model : iterable of iterable (np.ndarray or list of np.array)
        1D models (consisting of thicknesses and values) to plot
    ax : matplotlib axes [None - create new]
        axes object to plot in
    x : iterable
        positions of individual models
    cMin/cMax : float [None - autodetection from range]
        minimum and maximum colorscale range
    logScale : bool [True]
        use logarithmic color scaling
    zMin/zMax : float [0 - automatic]
        range for z (y axis) limits
    zLog : bool
        use logarithmic z (y axis) instead of linear
    topo : iterable
        vector of elevation for shifting
    thk : iterable
        vector of layer thicknesses for all models
    Returns
    -------
    ax : matplotlib axes [None - create new]
        axes object to plot in
    """
    if x is None:
        x = np.arange(len(models))

    topo = kwargs.pop('topo', np.zeros_like(x))

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    dxmed2 = np.median(np.diff(x)) / 2.
    patches = []
    zMinLimit = 9e99
    zMaxLimit = 0

    if thk is not None:
        nlay = len(models[0])
    else:
        nlay = int(np.floor((len(models[0]) + 1) / 2.))

    vals = np.zeros((len(models), nlay))
    for i, imod in enumerate(models):
        if thk is not None:  # take only resistivity from model
            vals[i, :] = imod
            thki = thk
        else:  # extract thickness from model vector
            if isinstance(imod, pg.Vector):
                vals[i, :] = imod[nlay - 1:2 * nlay - 1]
                thki = np.asarray(imod[:nlay - 1])
            else:
                vals[i, :] = imod[nlay - 1:2 * nlay - 1]
                thki = imod[:nlay - 1]

        if zMax > 0:
            z = np.hstack((0., np.cumsum(thki), zMax))
        else:
            thki = np.hstack((thki, thki[-1]*3))
            z = np.hstack((0., np.cumsum(thki)))

        z = topo[i] - z
        zMinLimit = min(zMinLimit, z[-1])
        zMaxLimit = max(zMaxLimit, z[0])

        for j in range(nlay):
            rect = Rectangle((x[i] - dxmed2, z[j]),
                             dxmed2 * 2, z[j+1]-z[j])
            patches.append(rect)

    p = PatchCollection(patches)  # , cmap=cmap, linewidths=0)
    if cMin is not None:
        p.set_clim(cMin, cMax)

    setMappableData(p, vals.ravel(), logScale=logScale)
    ax.add_collection(p)

    if logScale:
        norm = LogNorm(cMin, cMax)
        p.set_norm(norm)

    if 'cMap' in kwargs:
        p.set_cmap(kwargs['cMap'])

#    ax.set_ylim((zMaxLimit, zMin))
    ax.set_ylim((zMinLimit, zMaxLimit))

    if zLog:
        ax.set_yscale("log", nonposy='clip')

    ax.set_xlim((min(x) - dxmed2, max(x) + dxmed2))

    if title is not None:
        ax.set_title(title)
        
    if 'orientation' in kwargs:
        orientation = kwargs['orientation']

    if kwargs.pop('colorBar', True):
    #    cb = pg.viewer.mpl.createColorBar(p, cMin=cMin, cMax=cMax, nLevs=5, orientation=orientation)
        cb = plt.colorbar(p, orientation='vertical') #,aspect=50,pad=0.1)
        if 'cticks' in kwargs:
            xt = np.unique(np.clip(kwargs['cticks'], cMin, cMax))
            cb.set_ticks(xt)
            cb.set_ticklabels([str(xti) for xti in xt])
            
        if 'label' in kwargs:
            cb.set_label(kwargs['label'], fontsize=8)
            cb.ax.tick_params(labelsize=7)

    plt.draw()
    return ax  # maybe return cb as well?

def grid(model, depthmax=8, ny=101, nlay=2):
    """ Generates a grid from the model to plot a 2D section
    """
    # Arrays for plotting
    npos = np.shape(model)[0] # number of 1D models
   # ny = 101 # size of the grid in y direction
    y = np.linspace(0, depthmax, ny) # y axis [m]
    grid = np.zeros((npos, ny)) # empty grid
    thk = model[:,:nlay-1].copy() # define electrical conductivities
    sig = model[:,nlay-1:].copy()  # define thicknesses
    
    # Fill the grid with the conductivity values
    
    if nlay == 3:
        for i in range(npos):
            y1 = 0
            # First layer
            while y[y1] < thk[i,0]:
                grid[i, y1] = sig[i, 0]
                y1 += 1
                if y1 > ny-1:
                    break
                #y2 = y1
            # Second layer
            while y[y1] < (thk[i,0] + thk[i,1]):
                grid[i, y1] = sig[i, 1]
                y1 += 1
                if y1 > ny-1:
                    break
            # Third layer
            grid[i, y1:] = sig[i, 2]
    
    if nlay == 2:   
        for i in range(npos):
            y1 = 0
            # First layer
            while y[y1] < thk[i,0]:
                grid[i, y1] = sig[i, 0]
                y1 += 1
                if y1 > ny-1:
                    break
            while y[y1] >= thk[i,0]:
                grid[i, y1] = sig[i, 1]
                y1 += 1
                if y1 > ny-1:
                    break
        
    return grid

def plot_Data(data_true, data_est, ax, legen=False, ylab=False, ylabel = 'Q', xlab=False, fs=7):

    ax.semilogy(data_true[:,0], 'b', label='H2 true')
    ax.semilogy(data_true[:,1], '--b', label='H4 true')
    ax.semilogy(data_true[:,2], ':b', label='H8 true')
    ax.semilogy(data_true[:,3], 'k', label='V2 true')
    ax.semilogy(data_true[:,4], '--k', label='V4 true')
    ax.semilogy(data_true[:,5], ':k', label='V8 true')
    ax.semilogy(data_true[:,6], 'r', label='P2 true')
    ax.semilogy(data_true[:,7], '--r', label='P4 true')
    ax.semilogy(data_true[:,8], ':r', label='P8 true')

    ax.semilogy(data_est[:,0], '.b', label='H2 est')
    ax.semilogy(data_est[:,1], 'xb', label='H4 est')
    ax.semilogy(data_est[:,2], '^b', label='H8 est')
    ax.semilogy(data_est[:,3], '.k', label='V2 est')
    ax.semilogy(data_est[:,4], 'xk', label='V4 est')
    ax.semilogy(data_est[:,5], '^k', label='V8 est')
    ax.semilogy(data_est[:,6], '.r', label='P2 est')
    ax.semilogy(data_est[:,7], 'xr', label='P4 est')
    ax.semilogy(data_est[:,8], '^r', label='P8 est')
    
    if legen == True:
        ax.legend(bbox_to_anchor=(1, 1.05), fontsize=7)
        #ax.legend(fontsize=7, loc='upper right')
    
    if ylab == True:
        ax.set_ylabel(ylabel+' [PPT]', fontsize=fs)
    
    if xlab == True:
        ax.set_xlabel('Distance [m]', fontsize=fs)