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
from sklearn.metrics import root_mean_squared_error
from matplotlib import colors
import matplotlib.tri as tri

import pygimli as pg
from pygimli.utils import rndig
from pygimli.viewer.mpl import setMappableData
from pygimli.viewer.mpl import updateAxes as updateAxes_

import sys
sys.path.insert(1, '../../src')

# Import functions
from EM1D import EMf_3Lay_HVP, nrmse

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
    """ Function to plot data fit"""

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
    
    if ylab == True:
        ax.set_ylabel(ylabel+' [PPT]', fontsize=fs)
    
    if xlab == True:
        ax.set_xlabel('Distance [m]', fontsize=fs)
        
def Plot1DModel(model, depths, ax=None, model_name=None, model_style='k', ylab=False, xlab=False, lw=1):
    """ Function to plot a 1D model"""
    if ax is None:
        fig, ax = plt.subplots()
    fs=7
    ax.step(model, depths, model_style, label=model_name, linewidth=lw)
    ax.set_xscale('log')
    if xlab == True:
        ax.set_xlabel('$\sigma$ [mS/m]', fontsize=fs)
    if ylab == True:
        ax.set_ylabel('Depth [m]', fontsize=fs)
    if model_name is not None:
        ax.legend(fontsize=fs)
    ax.tick_params(labelsize=fs)
    plt.tight_layout()
    
    
def Plot_Noise_Analysis(models_noise, model_true, model_est, dmax=-8, ax=None, xlab=False, ylab=False):
    """ Function to plot Noise Analysis for 2-Layered case
    models_noise : models estimated in Noise Analysis
    model_true : true model
    model_est : estimated model without noise
    dmax : maximum depth for plot
    ax : axis
    ylab : y axis label
    xlab : x axis label
    """
    if ax is None:
        fig, ax = plt.subplots()
    fs=7
    for m in range(len(models_noise)):
        mod = models_noise[m]
        sigma_2Lay_plot = np.hstack((mod[1:], mod[-1]))
        depths_2Lay_plot = np.array([0, -mod[0], dmax])
        Plot1DModel(sigma_2Lay_plot, depths_2Lay_plot, ax=ax, model_style='paleturquoise')
    
    sigma_true = np.hstack((model_true[1:], model_true[-1]))
    depth_true = np.array([0, -model_true[0], dmax])
    Plot1DModel(sigma_true, depth_true, model_name='True', ax=ax, lw=4)
    
    sigma_est = np.hstack((model_est[1:], model_est[-1]))
    depth_est = np.array([0, -model_est[0], dmax])
    Plot1DModel(sigma_est, depth_est, model_name='Est', ax=ax, model_style='r', ylab=ylab, xlab=xlab)
    
# Function to plot Solution space for a fixed sigma1
def Plot_SolSpa_sigma1(fig, ax1, ax2, model, model_GS, model_GN, model_ini, model_hist,
                       err, models_err, pos, xmin=100, xmax=2000, ymin=0, ymax=7, case='', title='Solution Space', 
                       depthmax=-8, colorbar=True):
    """
    Plotting function of the solution space for a fixed sigma1
    
    ax1 : axis to plot the 1D model
    ax2 : axis to plot the solution space
    model : true model
    model_GS : model estimated GS
    model_GN : model estimated GN
    model_ini : initial model for Gauss-Newton inversion
    model_hist : update history of Gauss-Newton inversion
    err : nrmse values of the solution space
    models_err : models sampled in the solution space
    xmin : minimum x value for solution space plot in mS/m
    xmax : maximum x value for solution space plot in mS/m
    ymin : minimum y value for solution space plot in m
    ymax : maximum y value for solution space plot in m
    
    """
    # Depths to plot 1D models
    depth_true = np.array([0, -model[0], depthmax])
    depth_GS = np.array([0, -model_GS[0], depthmax])
    depth_GN = np.array([0, -model_GN[0], depthmax])
    depth_ini = np.array([0, -model_ini[0], depthmax])

    # Sigmas to plot 1D models in mS/m
    sigma_true = np.hstack([model[1:], model[-1]])*1000
    sigma_GS = np.hstack([model_GS[1:], model_GS[-1]])*1000
    sigma_GN = np.hstack([model_GN[1:], model_GN[-1]])*1000
    sigma_ini = np.hstack([model_ini[1:], model_ini[-1]])*1000
    
    # Plot 1D models
    ax1.step(sigma_true, depth_true, 'k', label = 'True', linewidth=4)
    ax1.step(sigma_GS, depth_GS, 'r', label='GS')
    ax1.step(sigma_GN, depth_GN, 'c', label='GI')
    ax1.step(sigma_ini, depth_ini, 'g', label='Initial')
    ax1.set_xlim([10,2000])
    ax1.set_ylabel('Depth [m]', fontsize=8)
    ax1.set_xlabel('$\sigma$ [mS/m]', fontsize=8)
    ax1.set_title('1D Model X=' +str(pos) + 'm - '+case, fontsize=8)
    ax1.set_xscale('log')
    ax1.legend(fontsize=7)
    ax1.tick_params(axis='both',labelsize=9)

    # Solution space values
    x = ((models_err[:,2])*1000) # conductivities of second layer in mS/m
    y = models_err[:,0]          # thicknesses of first layer
    z = err                      # nrmse values

    ngridx = 100
    ngridy = 200
    
    # Create grid values first.
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Plot solution space
    ax2.contour(xi, yi, zi*100, levels=15, linewidths=0.5, colors='k', )
    cntr1 = ax2.contourf(xi, yi, zi*100, levels=15, cmap="RdBu_r", vmin=0, vmax=40)
    ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax2.scatter(model[2]*1000, model[0], marker='o', c='w', label='True model', s=100)
    ax2.scatter(model_GS[2]*1000, model_GS[0], marker ='x', c='r', label='GS', s=100)
    ax2.scatter(model_GN[2]*1000, model_GN[0], marker ='.', c='y', label='GN', s=100)
    ax2.scatter(model_ini[2]*1000, model_ini[0], marker ='.', c='k', label='Initial', s=100)
    # Plot update path
    for i in range(len(model_hist)+1):
        x = model_hist[i-1:i+1,2]*1000
        y = model_hist[i-1:i+1,0]
        ax2.plot(x,y, ':k')
    ax2.set_xlabel('$\sigma_2$ [mS/m]', fontsize=8)
    ax2.set_ylabel('$h_1$ [m]', fontsize=8)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='both',labelsize=9)
    #ax2.set_title(method, fontsize=8)
    ax2.set_xscale('log')
    
    if colorbar==True:
        clb = fig.colorbar(cntr1, ax=ax2, ticks=[0, 10, 20, 30, 40])
        clb.ax.set_title('NRMSE %', fontsize=7)
        clb.ax.tick_params(labelsize=9)
    
# Function to plot Solution space for a fixed sigma2
def Plot_SolSpa_sigma2(fig, ax1, ax2, model, model_GS, model_GN, model_ini, model_hist,
                       err, models_err, pos, xmin=10, xmax=600, ymin=0, ymax=7, case='', title='Solution Space', 
                       depthmax=-8, colorbar=True):
    """
    Plotting function of the solution space for a fixed sigma2
    
    ax1 : axis to plot the 1D model
    ax2 : axis to plot the solution space
    model : true model
    model_GS : model estimated GS
    model_GN : model estimated GN
    model_ini : initial model for Gauss-Newton inversion
    model_hist : update history of Gauss-Newton inversion
    err : nrmse values of the solution space
    models_err : models sampled in the solution space
    xmin : minimum x value for solution space plot in mS/m
    xmax : maximum x value for solution space plot in mS/m
    ymin : minimum y value for solution space plot in m
    ymax : maximum y value for solution space plot in m
    
    """
    # Depths to plot 1D models
    depth_true = np.array([0, -model[0], depthmax])
    depth_GS = np.array([0, -model_GS[0], depthmax])
    depth_GN = np.array([0, -model_GN[0], depthmax])
    depth_ini = np.array([0, -model_ini[0], depthmax])

    # Sigmas to plot 1D models in mS/m
    sigma_true = np.hstack([model[1:], model[-1]])*1000
    sigma_GS = np.hstack([model_GS[1:], model_GS[-1]])*1000
    sigma_GN = np.hstack([model_GN[1:], model_GN[-1]])*1000
    sigma_ini = np.hstack([model_ini[1:], model_ini[-1]])*1000
    
    # Plot 1D models
    ax1.step(sigma_true, depth_true, 'k', label = 'True', linewidth=4)
    ax1.step(sigma_GS, depth_GS, 'r', label='GS')
    ax1.step(sigma_GN, depth_GN, 'c', label='GI')
    ax1.step(sigma_ini, depth_ini, 'g', label='Initial')
    ax1.set_xlim([10,2000])
    ax1.set_ylabel('Depth [m]', fontsize=8)
    ax1.set_xlabel('$\sigma$ [mS/m]', fontsize=8)
    ax1.set_title('1D Model X=' +str(pos) + 'm - '+case, fontsize=8)
    ax1.set_xscale('log')
    ax1.legend(fontsize=7)
    ax1.tick_params(axis='both',labelsize=9)

    # Solution space
    x = ((models_err[:,1])*1000) # conductivities of first layer in mS/m
    y = models_err[:,0]          # thicknesses of first layer in m
    z = err                      # nrmse values

    ngridx = 100
    ngridy = 200
    
    # Create grid values first.
    xi = np.linspace(np.min(x), np.max(x), ngridx)
    yi = np.linspace(np.min(y), np.max(y), ngridy)

    # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, z)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)

    # Plot solution space
    ax2.contour(xi, yi, zi*100, levels=15, linewidths=0.5, colors='k', )
    cntr1 = ax2.contourf(xi, yi, zi*100, levels=15, cmap="RdBu_r", vmin=0, vmax=40)
    ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax2.scatter(model[1]*1000, model[0], marker='o', c='w', label='True model', s=100)
    ax2.scatter(model_GS[1]*1000, model_GS[0], marker ='x', c='r', label='GS', s=100)
    ax2.scatter(model_GN[1]*1000, model_GN[0], marker ='.', c='y', label='GN', s=100)
    ax2.scatter(model_ini[1]*1000, model_ini[0], marker ='.', c='k', label='Initial', s=100)
    # Plot update history
    for i in range(len(model_hist)+1):
        x = model_hist[i-1:i+1,1]*1000
        y = model_hist[i-1:i+1,0]
        ax2.plot(x,y, ':k')
    ax2.set_xlabel('$\sigma_1$ [mS/m]', fontsize=8)
    ax2.set_ylabel('$h_1$ [m]', fontsize=8)
    ax2.legend(fontsize=7)
    ax2.tick_params(axis='both',labelsize=9)
    #ax2.set_title(method, fontsize=8)
    ax2.set_xscale('log')
    
    if colorbar==True:
        clb = fig.colorbar(cntr1, ax=ax2, ticks=[0, 10, 20, 30, 40])
        clb.ax.set_title('NRMSE %', fontsize=7)
        clb.ax.tick_params(labelsize=9)
        
# Function to plot model comparison Q + IP
def Plot_Models_2Lay_QIP(model_true, model_GS, model_GN, data_true, data_GS, data_GN, x, z1, case=''):

    # plotting options
    inputs1 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':False, 
        'fontsize':7, 
        'labelsize':7
    }
    
    inputs2 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':True, 
        'orientation':'vertical',
        'label': '$\sigma$ [mS/m]',
        'fontsize':7, 
        'labelsize':7
    }
    
    #fontsizes
    fs=8
    fs2=10
    
    fig, ax = plt.subplots(5,2,  sharex=True,  figsize=(7,7), layout='constrained')

    # True model
    showStitchedModels(model_true, ax = ax[0,0], **inputs1)
    fig.delaxes(ax[0,1])
    p = ax[0,0].get_position().get_points()
    x0, y0 = p[0]
    x1, y1 = p[1]
    ax[0,0].set_position([x0 + (x1)/2, y0+0.06, (x1-x0) , y1-y0])
    ax[0,0].set_title('True model', fontsize=fs2)
    ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0,0].tick_params(labelsize=fs)
    ax[0,0].text(-13,0.5, 'Case ' + case, fontsize='large')
    
    # Estimated models
    showStitchedModels(model_GS, ax = ax[1,0], **inputs1)
    ax[1,0].plot(x, z1, ':k')
    ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[1,0].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,1:], model_GS[:,1:]) + ' mS/m', fontsize=fs)
    ax[1,0].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,0], model_GS[:,0]) + ' m', fontsize=fs)
    ax[1,0].tick_params(labelsize=fs)
    ax[1,0].set_title('Estimated model - GS', fontsize=fs2)
    
    showStitchedModels(model_GN, ax = ax[1,1], **inputs2)
    ax[1,1].plot(x, z1, ':k')
    ax[1,1].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,1:], model_GN[:,1:]) + ' mS/m', fontsize=fs)
    ax[1,1].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,0], model_GN[:,0]) + ' m', fontsize=fs)  
    ax[1,1].tick_params(labelsize=fs)
    ax[1,1].set_title('Estimated model - GN', fontsize=fs2)
    
    # Calculate grids to plot relative difference
    model_true_grid = grid(model_true)
    model_GS_grid = grid(model_GS)
    model_GN_grid = grid(model_GN)
    
    # Calculate relative difference
    diff_GS = 100*np.abs(model_true_grid - model_GS_grid)/model_true_grid
    diff_GN = 100*np.abs(model_true_grid - model_GN_grid)/model_true_grid
    
    # Plot relative differences
    ax[2,0].imshow(diff_GS.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=0.1, vmax=100), cmap = 'binary', aspect='auto')
    ax[2,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[2,0].set_title('Relative difference % - GS', fontsize=fs2)
    ax[2,0].tick_params(labelsize=fs)
    ax[2,0].plot(x, z1, ':r')
    
    dc = ax[2,1].imshow(diff_GN.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=0.1, vmax=100), cmap = 'binary', aspect='auto')
    ax[2,1].set_title('Relative difference % - GN', fontsize=fs2)
    ax[2,1].tick_params(labelsize=fs)
    ax[2,1].plot(x, z1, ':r')
    cb = fig.colorbar(dc, ax=ax[2,1], location='right')
    cb.set_label('Rel. Difference %', fontsize=fs)
    cb.ax.tick_params(labelsize=7)
    
    # Plot data fit quadrature
    plot_Data(data_true, data_GS, ax=ax[3,0], ylab=True,)  
    ax[3,0].set_title('Quadrature - GS - RMSE: %2.3f' % root_mean_squared_error(data_true[:,:9], data_GS[:,:9]) + ' ppt', fontsize=fs2)
    ax[3,0].tick_params(labelsize=fs)
    
    plot_Data(data_true, data_GN, ax=ax[3,1], )
    ax[3,1].set_title('Quadrature - GN - RMSE: %2.3f' % root_mean_squared_error(data_true[:,:9], data_GN[:,:9]) + ' ppt', fontsize=fs2)
    ax[3,1].tick_params(labelsize=fs)
    
    # Plot data fit in-phase    
    plot_Data(data_true[:, 9:], data_GS[:, 9:], ax=ax[4,0], ylab=True, ylabel='IP')
    ax[4,0].set_xlabel('Distance [m]', fontsize=fs)
    ax[4,0].set_title('In Phase - GS - RMSE: %2.3f' % root_mean_squared_error(data_true[:,9:], data_GS[:,9:]) + ' ppt', fontsize=fs2)
    ax[4,0].tick_params(labelsize=fs)
 
    plot_Data(data_true[:, 9:], data_GN[:, 9:], ax=ax[4,1],)
    ax[4,1].set_xlabel('Distance [m]', fontsize=fs)
    ax[4,1].set_title('In Phase - GN - RMSE: %2.3f' % root_mean_squared_error(data_true[:,9:], data_GN[:,9:]) + ' ppt', fontsize=fs2)
    ax[4,1].tick_params(labelsize=fs)

    handles, labels = ax[3,1].get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='lower right', fontsize=fs, bbox_to_anchor=(1.03, 0) ) 
    
# Function for only one data component (Q or IP)
def Plot_Models_2Lay(model_true, model_GS, model_GN, data_true, data_GS, data_GN, x, z1, data_type = '', case='' ):

    # Plotting options
    inputs1 = {
        'colorBar':False,
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'labelsize':7
    }

    inputs2 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':True, 
        'orientation':'vertical',
        'label': '$\sigma$ [mS/m]',
        'fontsize':7, 
        'labelsize':7
    }
    
    # fontsizes
    fs=8
    fs2=10
    
    fig, ax = plt.subplots(3,2,  sharex=True, figsize=(7,5))
    fig.suptitle('Case ' +case)

    # Plot estimated models
    showStitchedModels(model_GS, ax = ax[0,0], **inputs2)
    ax[0,0].plot(x, z1, ':k')
    ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0,0].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,1:], model_GS[:,1:]) + ' mS/m', fontsize=fs)
    ax[0,0].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,0], model_GS[:,0]) + ' m', fontsize=fs)
    ax[0,0].tick_params(labelsize=fs)
    ax[0,0].set_title('Estimated model GS', fontsize=fs2)
    
    showStitchedModels(model_GN, ax = ax[0,1], **inputs2)
    ax[0,1].plot(x, z1, ':k')
    ax[0,1].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,1:], model_GN[:,1:]) + ' mS/m', fontsize=fs)
    ax[0,1].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,0], model_GN[:,0]) + ' m', fontsize=fs)  
    ax[0,1].tick_params(labelsize=fs)
    ax[0,1].set_title('Estimated model GN', fontsize=fs2)
    
    # Calculate a grid of the estimated models to compute relative differences
    model_true_grid = grid(model_true)
    model_GS_grid = grid(model_GS)
    model_GN_grid = grid(model_GN)
    
    # Calculate relative differences
    diff_GS = 100*np.abs(model_true_grid - model_GS_grid)/model_true_grid
    diff_GN = 100*np.abs(model_true_grid - model_GN_grid)/model_true_grid
    
    # Plot relative differences
    dc = ax[1,0].imshow(diff_GS.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,0].plot(x, z1, ':r')
    ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[1,0].set_title('Relative difference % - GS', fontsize=fs2)
    ax[1,0].tick_params(labelsize=fs)
    cb = fig.colorbar(dc, ax=ax[1,0], location='right', )
    cb.set_label('Rel. Difference %', fontsize=fs)
    cb.ax.tick_params(labelsize=fs)
    
    dc = ax[1,1].imshow(diff_GN.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=0.1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,1].plot(x, z1, ':r')
    ax[1,1].set_title('Relative difference % - GN', fontsize=fs2)
    ax[1,1].tick_params(labelsize=fs)
    cb = fig.colorbar(dc, ax=ax[1,1], location='right', )
    cb.set_label('Rel. Difference %', fontsize=fs)
    cb.ax.tick_params(labelsize=fs)

    if data_type == 'Quadrature':
        yl = 'Q'
    else:
        yl = 'IP'

    # plot data fit
    plot_Data(data_true, data_GS, ax=ax[2,0], ylab=True, ylabel=yl)
    ax[2,0].set_title(data_type + ' - GS - RMSE: %2.3f' % root_mean_squared_error(data_true, data_GS) + ' ppt', fontsize=fs2)
    ax[2,0].set_xlabel('Distance [m]', fontsize=fs)
    ax[2,0].tick_params(labelsize=fs)
    
    plot_Data(data_true, data_GN, ax=ax[2,1],)
    ax[2,1].set_title(data_type +' - GN - RMSE: %2.3f' % root_mean_squared_error(data_true, data_GN) + ' ppt', fontsize=fs2)
    ax[2,1].set_xlabel('Distance [m]', fontsize=fs)
    ax[2,1].tick_params(labelsize=fs)
    
    plt.tight_layout()

    handles, labels = ax[2,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=6, bbox_to_anchor=(0.5, -0.1) ) 
    
# Plot models estimation with increasing noise in data
def Plot_Models_Noise(model_true, model_GS_n2, model_GS_n5, model_GS_n10, 
                      model_GN_n2, model_GN_n5, model_GN_n10,
                      data_true, data_GS_n2, data_GS_n5, data_GS_n10, 
                      data_GN_n2, data_GN_n5, data_GN_n10, x, z1):
    
    # Plotting options
    inputs1 = {
        'colorBar':False,
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
    }

    # fontsizes
    fs=8
    fs2=10
    
    fig, ax = plt.subplots(8,3, sharex=True, figsize = (9,12))

    # Plot estimated models
    showStitchedModels(model_GS_n2, ax=ax[0,0], **inputs1)
    ax[0,0].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,1:], model_GS_n2[:,1:]) + ' mS/m', fontsize = fs)
    ax[0,0].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,0], model_GS_n2[:,0]) + ' m', fontsize = fs)
    ax[0,0].set_title('Estimated model GS - Noise 2.5%', fontsize=fs2)
    ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0,0].tick_params(labelsize=fs)
    ax[0,0].plot(x, z1, ':k')
    
    showStitchedModels(model_GS_n5, ax=ax[0,1], **inputs1)
    ax[0,1].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,1:], model_GS_n5[:,1:]) + ' mS/m', fontsize = fs)
    ax[0,1].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,0], model_GS_n5[:,0]) + ' m', fontsize = fs)
    ax[0,1].set_title('Estimated model GS - Noise 5%', fontsize=fs2)
    ax[0,1].tick_params(labelsize=fs)
    ax[0,1].plot(x, z1, ':k')
    
    showStitchedModels(model_GS_n10, ax=ax[0,2], **inputs1)
    ax[0,2].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,1:], model_GS_n10[:,1:]) + ' mS/m', fontsize = fs)
    ax[0,2].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,0], model_GS_n10[:,0]) + ' m', fontsize = fs)
    ax[0,2].set_title('Estimated model GS - Noise 10%', fontsize=fs2)
    ax[0,2].tick_params(labelsize=fs)
    ax[0,2].plot(x, z1, ':k')
    
    showStitchedModels(model_GN_n2, ax=ax[4,0], **inputs1)
    ax[4,0].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,1:], model_GN_n2[:,1:]) + ' mS/m', fontsize = fs)
    ax[4,0].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,0], model_GN_n2[:,0]) + ' m', fontsize = fs)
    ax[4,0].set_title('Estimated model GN - Noise 2.5%', fontsize=fs2)
    ax[4,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[4,0].tick_params(labelsize=fs)
    ax[4,0].plot(x, z1, ':k')
    
    showStitchedModels(model_GN_n5, ax=ax[4,1], **inputs1)
    ax[4,1].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,1:], model_GN_n5[:,1:]) + ' mS/m', fontsize = fs)
    ax[4,1].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,0], model_GN_n5[:,0]) + ' m', fontsize = fs)
    ax[4,1].set_title('Estimated model GN - Noise 5%', fontsize=fs2)
    ax[4,1].tick_params(labelsize=fs)
    ax[4,1].plot(x, z1, ':k')
    
    showStitchedModels(model_GN_n10, ax=ax[4,2], **inputs1)
    ax[4,2].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,1:], model_GN_n10[:,1:]) + ' mS/m', fontsize = fs)
    ax[4,2].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,0], model_GN_n10[:,0]) + ' m', fontsize = fs)
    ax[4,2].set_title('Estimated model GN - Noise 10%', fontsize=fs2)
    ax[4,2].tick_params(labelsize=fs)
    ax[4,2].plot(x, z1, ':k')
    
    # Calculate grids from the estimated models to compute relative differences
    model_true_grid = grid(model_true)
    model_GS_n2_grid = grid(model_GS_n2)
    model_GS_n5_grid = grid(model_GS_n5)
    model_GS_n10_grid = grid(model_GS_n10)
    model_GN_n2_grid = grid(model_GN_n2)
    model_GN_n5_grid = grid(model_GN_n5)
    model_GN_n10_grid = grid(model_GN_n10)
    
    # Calculate relative differences
    diff_GS_n2 = 100*np.abs(model_true_grid - model_GS_n2_grid)/model_true_grid
    diff_GS_n5 = 100*np.abs(model_true_grid - model_GS_n5_grid)/model_true_grid
    diff_GS_n10 = 100*np.abs(model_true_grid - model_GS_n10_grid)/model_true_grid
    
    diff_GN_n2 = 100*np.abs(model_true_grid - model_GN_n2_grid)/model_true_grid
    diff_GN_n5 = 100*np.abs(model_true_grid - model_GN_n5_grid)/model_true_grid
    diff_GN_n10 = 100*np.abs(model_true_grid - model_GN_n10_grid)/model_true_grid
    
    # Plot relative differences    
    ax[1,0].imshow(diff_GS_n2.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,0].set_ylabel('Rel diff %', fontsize=fs)
    ax[1,0].tick_params(labelsize=fs)
    ax[1,0].plot(x, z1, ':r')
    ax[1,0].set_title('Rel. difference % - GS', fontsize=fs2)
    ax[1,1].imshow(diff_GS_n5.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,1].tick_params(labelsize=fs)
    ax[1,1].plot(x, z1, ':r')
    ax[1,1].set_title('Rel. difference % - GS', fontsize=fs2)
    ax[1,2].imshow(diff_GS_n10.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,2].tick_params(labelsize=fs)
    ax[1,2].set_title('Rel. difference % - GS', fontsize=fs2)
    ax[1,2].plot(x, z1, ':r')
    
    ax[5,0].imshow(diff_GN_n2.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[5,0].set_ylabel('Rel diff %', fontsize=fs)
    ax[5,0].tick_params(labelsize=fs)
    ax[5,0].plot(x, z1, ':r')
    ax[5,0].set_title('Rel. difference % - GN', fontsize=fs2)
    ax[5,1].imshow(diff_GN_n5.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[5,1].tick_params(labelsize=fs)
    ax[5,1].set_title('Rel. difference % - GN', fontsize=fs2)
    ax[5,1].plot(x, z1, ':r')
    ax[5,2].imshow(diff_GN_n10.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[5,2].tick_params(labelsize=fs)
    ax[5,2].set_title('Rel. difference % - GN', fontsize=fs2)
    ax[5,2].plot(x, z1, ':r')
    
    # Plot data fit quadrature    
    plot_Data(data_true[:,:9], data_GS_n2[:,:9], ax=ax[2,0])
    ax[2,0].set_title('Quadrature - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GS_n2[:,:9]) + ' ppt', fontsize=fs2)
    ax[2,0].set_ylabel('[ppt]', fontsize=fs)
    ax[2,0].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GS_n5[:,:9], ax=ax[2,1])
    ax[2,1].set_title('Quadrature - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GS_n5[:,:9]) + ' ppt', fontsize=fs2)
    ax[2,1].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GS_n10[:,:9], ax=ax[2,2])
    ax[2,2].set_title('Quadrature - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GS_n10[:,:9]) + ' ppt', fontsize=fs2)
    ax[2,2].tick_params(labelsize=fs)
    
    plot_Data(data_true[:,:9], data_GN_n2[:,:9], ax=ax[6,0])
    ax[6,0].set_title('Quadrature - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GN_n2[:,:9]) + ' ppt', fontsize=10)
    ax[6,0].set_ylabel('[ppt]', fontsize=fs)
    ax[6,0].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GN_n5[:,:9], ax=ax[6,1])
    ax[6,1].set_title('Quadrature - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GN_n5[:,:9]) + ' ppt', fontsize=10)
    ax[6,1].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GN_n10[:,:9], ax=ax[6,2])
    ax[6,2].set_title('Quadrature - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GN_n10[:,:9]) + ' ppt', fontsize=10)
    ax[6,2].tick_params(labelsize=fs)

    # Plot data fit in-phase
    plot_Data(data_true[:,9:], data_GS_n2[:,9:], ax=ax[3,0])
    ax[3,0].set_ylabel('[ppt]', fontsize=fs)
    ax[3,0].set_title('In Phase - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GS_n2[:,9:]) + ' ppt', fontsize=fs2)
    ax[3,0].tick_params(labelsize=fs)
    plot_Data(data_true[:,9:], data_GS_n5[:,9:], ax=ax[3,1])
    ax[3,1].set_title('In Phase - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GS_n5[:,9:]) + ' ppt', fontsize=fs2)
    ax[3,1].tick_params(labelsize=fs)
    plot_Data(data_true[:,9:], data_GS_n10[:,9:], ax=ax[3,2])
    ax[3,2].set_title('In Phase - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GS_n10[:,9:]) + ' ppt', fontsize=fs2)
    ax[3,2].tick_params(labelsize=fs)
    
    plot_Data(data_true[:,9:], data_GN_n2[:,9:], ax=ax[7,0])
    ax[7,0].set_ylabel('[ppt]', fontsize=fs)
    ax[7,0].set_title('In Phase - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GN_n2[:,9:]) + ' ppt', fontsize=10)
    ax[7,0].tick_params(labelsize=fs)
    ax[7,0].set_xlabel('Distance [m]', fontsize=fs)
    plot_Data(data_true[:,9:], data_GN_n5[:,9:], ax=ax[7,1])
    ax[7,1].set_title('In Phase - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GN_n5[:,9:]) + ' ppt', fontsize=10)
    ax[7,1].tick_params(labelsize=fs)
    ax[7,1].set_xlabel('Distance [m]', fontsize=fs)
    plot_Data(data_true[:,9:], data_GN_n10[:,9:], ax=ax[7,2])
    ax[7,2].set_title('In Phase - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GN_n10[:,9:]) + ' ppt', fontsize=10)
    ax[7,2].tick_params(labelsize=fs)
    ax[7,2].set_xlabel('Distance [m]', fontsize=fs)

    plt.tight_layout()

def Plot_Models_3Lay_QIP(model_true, model_GS, model_GN, data_true, data_GS, data_GN, x, z1, z2, case=''):

    inputs1 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':False, 
    }

    inputs2 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':True, 
        'orientation':'vertical',
        'label': '$\sigma$ [mS/m]',
        'fontsize':7, 
        'labelsize':7
    }
    
    fs=8
    fs2=10
    fig, ax = plt.subplots(5,2,  sharex=True,  figsize=(7,7), layout='constrained')

    showStitchedModels(model_true, ax = ax[0,0], **inputs1)

    fig.delaxes(ax[0,1])
    p = ax[0,0].get_position().get_points()
    x0, y0 = p[0]
    x1, y1 = p[1]
    ax[0,0].set_position([x0 + (x1)/2, y0+0.06, (x1-x0) , y1-y0])
    ax[0,0].set_title('True model', fontsize=fs2)
    ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0,0].tick_params(labelsize=fs)
    ax[0,0].text(-10,0.5, 'Case ' + case +' [Q+IP]')
    
    showStitchedModels(model_GS, ax = ax[1,0], **inputs1)
    ax[1,0].plot(x,-z1,':k')
    ax[1,0].plot(x,-z1-z2,':k')
    ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[1,0].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GS[:,2:]) + ' mS/m', fontsize=fs)
    ax[1,0].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GS[:,:2]) + ' m', fontsize=fs)
    ax[1,0].tick_params(labelsize=fs)
    ax[1,0].set_title('Estimated model - GS', fontsize=fs2)
    
    showStitchedModels(model_GN, ax = ax[1,1], **inputs2)
    #ax[1,1].set_ylabel('Depth [m]')
    ax[1,1].plot(x,-z1,':k')
    ax[1,1].plot(x,-z1-z2,':k')
    ax[1,1].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GN[:,2:]) + ' mS/m', fontsize=fs)
    ax[1,1].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GN[:,:2]) + ' m', fontsize=fs)  
    ax[1,1].tick_params(labelsize=fs)
    ax[1,1].set_title('Estimated model - GN', fontsize=fs2)
    
    model_true_grid = grid(model_true, nlay=3)
    model_GS_grid = grid(model_GS, nlay=3)
    model_GN_grid = grid(model_GN, nlay=3)
    
    diff_GS = 100*np.abs(model_true_grid - model_GS_grid)/model_true_grid
    diff_GN = 100*np.abs(model_true_grid - model_GN_grid)/model_true_grid
    
    ax[2,0].imshow(diff_GS.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=0.1, vmax=100), cmap = 'binary', aspect='auto')
    ax[2,0].plot(x,-z1,':r')
    ax[2,0].plot(x,-z1-z2,':r')
    ax[2,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[2,0].set_title('Relative difference % - GS', fontsize=fs2)
    ax[2,0].tick_params(labelsize=fs)
    
    dc = ax[2,1].imshow(diff_GN.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=0.1, vmax=100), cmap = 'binary', aspect='auto')
    ax[2,1].plot(x,-z1,':r')
    ax[2,1].plot(x,-z1-z2,':r')
    ax[2,1].set_title('Relative difference % - GN', fontsize=fs2)
    ax[2,1].tick_params(labelsize=fs)
    cb = fig.colorbar(dc, ax=ax[2,1], location='right', )#pad=1e-9)
    cb.set_label('Rel. Difference %', fontsize=fs)
    cb.ax.tick_params(labelsize=7)

    plot_Data(data_true, data_GS, ax=ax[3,0], ylab=True,)  
    ax[3,0].set_title('Quadrature - GS - RMSE: %2.3f' % root_mean_squared_error(data_true[:,:9], data_GS[:,:9]) + ' ppt', fontsize=fs2)
    ax[3,0].tick_params(labelsize=fs)
    
    plot_Data(data_true, data_GN, ax=ax[3,1], )
    ax[3,1].set_title('Quadrature - GN - RMSE: %2.3f' % root_mean_squared_error(data_true[:,:9], data_GN[:,:9]) + ' ppt', fontsize=fs2)
    ax[3,1].tick_params(labelsize=fs)
    
    plot_Data(data_true[:, 9:], data_GS[:, 9:], ax=ax[4,0], ylab=True, ylabel='IP')
    ax[4,0].set_xlabel('Distance [m]', fontsize=fs)
    ax[4,0].set_title('In Phase - GS - RMSE: %2.3f' % root_mean_squared_error(data_true[:,9:], data_GS[:,9:]) + ' ppt', fontsize=fs2)
    ax[4,0].tick_params(labelsize=fs)
 
    plot_Data(data_true[:, 9:], data_GN[:, 9:], ax=ax[4,1], )
    ax[4,1].set_xlabel('Distance [m]', fontsize=fs)
    ax[4,1].set_title('In Phase - GN - RMSE: %2.3f' % root_mean_squared_error(data_true[:,9:], data_GN[:,9:]) + ' ppt', fontsize=fs2)
    ax[4,1].tick_params(labelsize=fs)

    handles, labels = ax[3,1].get_legend_handles_labels()
    
    fig.legend(handles, labels, loc='lower right', fontsize=fs, bbox_to_anchor=(1.03, 0) ) 
    
def Plot_Models_3Lay(model_true, model_GS, model_GN, data_true, data_GS, data_GN, x, z1, z2, data_type = '', case='' ):

    inputs1 = {
        'colorBar':False,
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'labelsize':7
    }

    inputs2 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':True, 
        'orientation':'vertical',
        'label': '$\sigma$ [mS/m]',
        'fontsize':7, 
        'labelsize':7
    }
    
    fs=8
    fs2=10
    fig, ax = plt.subplots(3,2,  sharex=True,)

    showStitchedModels(model_GS, ax = ax[0,0], **inputs1)
    ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0,0].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GS[:,2:]) + ' mS/m', fontsize=fs)
    ax[0,0].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GS[:,:2]) + ' m', fontsize=fs)
    ax[0,0].tick_params(labelsize=fs)
    ax[0,0].set_title('Estimated model GS', fontsize=fs2)
    ax[0,0].text(-4,0.7, 'Case ' + case)
    ax[0,0].plot(x,-z1,':k')
    ax[0,0].plot(x,-z1-z2,':k')
    
    model_true_grid = grid(model_true, nlay=3)
    model_GS_grid = grid(model_GS, nlay=3)
    model_GN_grid = grid(model_GN, nlay=3)
    
    diff_GS = 100*np.abs(model_true_grid - model_GS_grid)/model_true_grid
    diff_GN = 100*np.abs(model_true_grid - model_GN_grid)/model_true_grid
    
    ax[1,0].imshow(diff_GS.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[1,0].set_title('Relative difference % - GS', fontsize=fs2)
    ax[1,0].tick_params(labelsize=fs)
    ax[1,0].plot(x,-z1,':r')
    ax[1,0].plot(x,-z1-z2,':r')
    
    dc = ax[1,1].imshow(diff_GN.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=0.1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,1].set_title('Relative difference % - GN', fontsize=fs2)
    ax[1,1].tick_params(labelsize=fs)
    ax[1,1].plot(x,-z1,':r')
    ax[1,1].plot(x,-z1-z2,':r')
    #cb = fig.colorbar(dc, ax=ax[1,1], location='right', )#pad=1e-9)
    #cb.set_label('Rel. Difference %', fontsize=fs)
    #cb.ax.tick_params(labelsize=fs)

    if data_type == 'Quadrature':
        yl = 'Q'
    else:
        yl = 'IP'

    plot_Data(data_true, data_GS, ax=ax[2,0], ylab=True, ylabel=yl)
    ax[2,0].set_title(data_type + ' - GS - RMSE: %2.3f' % root_mean_squared_error(data_true, data_GS) + ' ppt', fontsize=fs2)
    ax[2,0].set_xlabel('Distance [m]', fontsize=fs)
    ax[2,0].tick_params(labelsize=fs)
    
    showStitchedModels(model_GN, ax = ax[0,1], **inputs1)
    ax[0,1].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GN[:,2:]) + ' mS/m', fontsize=fs)
    ax[0,1].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GN[:,:2]) + ' m', fontsize=fs)  
    ax[0,1].tick_params(labelsize=fs)
    ax[0,1].set_title('Estimated model GN', fontsize=fs2)
    ax[0,1].plot(x,-z1,':k')
    ax[0,1].plot(x,-z1-z2,':k')
    
    plot_Data(data_true, data_GN, ax=ax[2,1],)
    ax[2,1].set_title(data_type +' - GN - RMSE: %2.3f' % root_mean_squared_error(data_true, data_GN) + ' ppt', fontsize=fs2)
    ax[2,1].set_xlabel('Distance [m]', fontsize=fs)
    ax[2,1].tick_params(labelsize=fs)
    
    plt.tight_layout()

    handles, labels = ax[2,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', fontsize=fs, ncol=6, bbox_to_anchor=(0.5, -0.1) ) 
    
def Plot_Noise_3Lay(model_true, model_GS_n2, model_GS_n5, model_GS_n10, model_GN_n2, model_GN_n5, model_GN_n10,
                    data_true, data_GS_n2, data_GS_n5, data_GS_n10, data_GN_n2, data_GN_n5, data_GN_n10, x, z1, z2):
    inputs1 = {
        'colorBar':False,
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
    }

    inputs2 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':True, 
        'orientation':'vertical',
        'label': 'Elec. cond. [mS/m]'
    }

    fs=8
    fs2=10
    fig, ax = plt.subplots(8,3, sharex=True, figsize = (9,12))

    showStitchedModels(model_GS_n2, ax=ax[0,0], **inputs1)
    ax[0,0].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,2:], model_GS_n2[:,2:]) + ' mS/m', fontsize = fs)
    ax[0,0].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,:2], model_GS_n2[:,:2]) + ' m', fontsize = fs)
    ax[0,0].set_title('Estimated model GS - Noise 2.5%', fontsize=fs2)
    ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0,0].tick_params(labelsize=fs)
    ax[0,0].plot(x, -z1, ':k')
    ax[0,0].plot(x, -z1-z2, ':k')
    
    showStitchedModels(model_GS_n5, ax=ax[0,1], **inputs1)
    ax[0,1].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,2:], model_GS_n5[:,2:]) + ' mS/m', fontsize = fs)
    ax[0,1].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,:2], model_GS_n5[:,:2]) + ' m', fontsize = fs)
    ax[0,1].set_title('Estimated model GS - Noise 5%', fontsize=fs2)
    ax[0,1].tick_params(labelsize=fs)
    ax[0,1].plot(x, -z1, ':k')
    ax[0,1].plot(x, -z1-z2, ':k')
    
    showStitchedModels(model_GS_n10, ax=ax[0,2], **inputs1)
    ax[0,2].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,2:], model_GS_n10[:,2:]) + ' mS/m', fontsize = fs)
    ax[0,2].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,:2], model_GS_n10[:,:2]) + ' m', fontsize = fs)
    ax[0,2].set_title('Estimated model GS - Noise 10%', fontsize=fs2)
    ax[0,2].tick_params(labelsize=fs)
    ax[0,2].plot(x, -z1, ':k')
    ax[0,2].plot(x, -z1-z2, ':k')
    
    # Rel diff  
    model_true_grid = grid(model_true, nlay=3)
    model_GS_n2_grid = grid(model_GS_n2, nlay=3)
    model_GS_n5_grid = grid(model_GS_n5, nlay=3)
    model_GS_n10_grid = grid(model_GS_n10, nlay=3)
    model_GN_n2_grid = grid(model_GN_n2, nlay=3)
    model_GN_n5_grid = grid(model_GN_n5, nlay=3)
    model_GN_n10_grid = grid(model_GN_n10, nlay=3)
    
    diff_GS_n2 = 100*np.abs(model_true_grid - model_GS_n2_grid)/model_true_grid
    diff_GS_n5 = 100*np.abs(model_true_grid - model_GS_n5_grid)/model_true_grid
    diff_GS_n10 = 100*np.abs(model_true_grid - model_GS_n10_grid)/model_true_grid
    
    diff_GN_n2 = 100*np.abs(model_true_grid - model_GN_n2_grid)/model_true_grid
    diff_GN_n5 = 100*np.abs(model_true_grid - model_GN_n5_grid)/model_true_grid
    diff_GN_n10 = 100*np.abs(model_true_grid - model_GN_n10_grid)/model_true_grid
    
    ax[1,0].imshow(diff_GS_n2.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,0].set_ylabel('Rel diff %', fontsize=fs)
    ax[1,0].tick_params(labelsize=fs)
    ax[1,0].plot(x, -z1, ':r')
    ax[1,0].plot(x, -z1-z2, ':r')
    ax[1,0].set_title('Rel. difference % - GS', fontsize=fs2)
    ax[1,1].imshow(diff_GS_n5.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,1].tick_params(labelsize=fs)
    ax[1,1].plot(x, -z1, ':r')
    ax[1,1].plot(x, -z1-z2, ':r')
    ax[1,1].set_title('Rel. difference % - GS', fontsize=fs2)
    ax[1,2].imshow(diff_GS_n10.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[1,2].tick_params(labelsize=fs)
    ax[1,2].set_title('Rel. difference % - GS', fontsize=fs2)
    ax[1,2].plot(x, -z1, ':r')
    ax[1,2].plot(x, -z1-z2, ':r')
    
    plot_Data(data_true[:,:9], data_GS_n2[:,:9], ax=ax[2,0])
    ax[2,0].set_title('Quadrature - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GS_n2[:,:9]) + ' ppt', fontsize=fs2)
    ax[2,0].set_ylabel('[ppt]', fontsize=fs)
    ax[2,0].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GS_n5[:,:9], ax=ax[2,1])
    ax[2,1].set_title('Quadrature - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GS_n5[:,:9]) + ' ppt', fontsize=fs2)
    ax[2,1].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GS_n10[:,:9], ax=ax[2,2])
    ax[2,2].set_title('Quadrature - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GS_n10[:,:9]) + ' ppt', fontsize=fs2)
    ax[2,2].tick_params(labelsize=fs)

    plot_Data(data_true[:,9:], data_GS_n2[:,9:], ax=ax[3,0])
    ax[3,0].set_ylabel('[ppt]', fontsize=fs)
    ax[3,0].set_title('In Phase - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GS_n2[:,9:]) + ' ppt', fontsize=fs2)
    ax[3,0].tick_params(labelsize=fs)
    plot_Data(data_true[:,9:], data_GS_n5[:,9:], ax=ax[3,1])
    ax[3,1].set_title('In Phase - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GS_n5[:,9:]) + ' ppt', fontsize=fs2)
    ax[3,1].tick_params(labelsize=fs)
    plot_Data(data_true[:,9:], data_GS_n10[:,9:], ax=ax[3,2])
    ax[3,2].set_title('In Phase - GS - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GS_n10[:,9:]) + ' ppt', fontsize=fs2)
    ax[3,2].tick_params(labelsize=fs)
    
    showStitchedModels(model_GN_n2, ax=ax[4,0], **inputs1)
    ax[4,0].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,2:], model_GN_n2[:,2:]) + ' mS/m', fontsize = fs)
    ax[4,0].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,:2], model_GN_n2[:,:2]) + ' m', fontsize = fs)
    ax[4,0].set_title('Estimated model GN - Noise 2.5%', fontsize=fs2)
    ax[4,0].set_ylabel('Depth [m]', fontsize=fs)
    ax[4,0].tick_params(labelsize=fs)
    ax[4,0].plot(x, -z1, ':k')
    ax[4,0].plot(x, -z1-z2, ':k')
    
    showStitchedModels(model_GN_n5, ax=ax[4,1], **inputs1)
    ax[4,1].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,2:], model_GN_n5[:,2:]) + ' mS/m', fontsize = fs)
    ax[4,1].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,:2], model_GN_n5[:,:2]) + ' m', fontsize = fs)
    ax[4,1].set_title('Estimated model GN - Noise 5%', fontsize=fs2)
    ax[4,1].tick_params(labelsize=fs)
    ax[4,1].plot(x, -z1, ':k')
    ax[4,1].plot(x, -z1-z2, ':k')
    
    showStitchedModels(model_GN_n10, ax=ax[4,2], **inputs1)
    ax[4,2].text(0,-6, 'RMSE $\sigma$: %2.2f' % root_mean_squared_error(model_true[:,2:], model_GN_n10[:,2:]) + ' mS/m', fontsize = fs)
    ax[4,2].text(0,-7, 'RMSE $h$: %2.2f' % root_mean_squared_error(model_true[:,:2], model_GN_n10[:,:2]) + ' m', fontsize = fs)
    ax[4,2].set_title('Estimated model GN - Noise 10%', fontsize=fs2)
    ax[4,2].tick_params(labelsize=fs)
    ax[4,2].plot(x, -z1, ':k')
    ax[4,2].plot(x, -z1-z2, ':k')

    # Rel diff
    ax[5,0].imshow(diff_GN_n2.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[5,0].set_ylabel('Rel diff %', fontsize=fs)
    ax[5,0].tick_params(labelsize=fs)
    ax[5,0].set_title('Rel. difference % - Opt', fontsize=fs2)
    ax[5,1].imshow(diff_GN_n5.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[5,1].tick_params(labelsize=fs)
    ax[5,1].set_title('Rel. difference % - Opt', fontsize=fs2)
    ax[5,2].imshow(diff_GN_n10.T, extent=[-1,20, -8,0], norm=colors.LogNorm(vmin=1, vmax=100), cmap = 'binary', aspect='auto')
    ax[5,2].tick_params(labelsize=fs)
    ax[5,2].set_title('Rel. difference % - Opt', fontsize=fs2)
    
    plot_Data(data_true[:,:9], data_GN_n2[:,:9], ax=ax[6,0])
    ax[6,0].set_title('Quadrature - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GN_n2[:,:9]) + ' ppt', fontsize=10)
    ax[6,0].set_ylabel('[ppt]', fontsize=fs)
    ax[6,0].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GN_n5[:,:9], ax=ax[6,1])
    ax[6,1].set_title('Quadrature - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GN_n5[:,:9]) + ' ppt', fontsize=10)
    ax[6,1].tick_params(labelsize=fs)
    plot_Data(data_true[:,:9], data_GN_n10[:,:9], ax=ax[6,2])
    ax[6,2].set_title('Quadrature - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,:9], data_GN_n10[:,:9]) + ' ppt', fontsize=10)
    ax[6,2].tick_params(labelsize=fs)

    plot_Data(data_true[:,9:], data_GN_n2[:,9:], ax=ax[7,0])
    ax[7,0].set_ylabel('[ppt]', fontsize=fs)
    ax[7,0].set_title('In Phase - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GN_n2[:,9:]) + ' ppt', fontsize=10)
    ax[7,0].tick_params(labelsize=fs)
    ax[7,0].set_xlabel('Distance [m]', fontsize=fs)
    plot_Data(data_true[:,9:], data_GN_n5[:,9:], ax=ax[7,1])
    ax[7,1].set_title('In Phase - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GN_n5[:,9:]) + ' ppt', fontsize=10)
    ax[7,1].tick_params(labelsize=fs)
    ax[7,1].set_xlabel('Distance [m]', fontsize=fs)
    plot_Data(data_true[:,9:], data_GN_n10[:,9:], ax=ax[7,2])
    ax[7,2].set_title('In Phase - GN - RMSE: %2.2f' % root_mean_squared_error(data_true[:,9:], data_GN_n10[:,9:]) + ' ppt', fontsize=10)
    ax[7,2].tick_params(labelsize=fs)
    ax[7,2].set_xlabel('Distance [m]', fontsize=fs)

    plt.tight_layout()
    
def Plot_Models_GSGN(model_true, model_GS, model_GN, model_GSGN, x, z1, z2, case=''):

    inputs1 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':False, 
    }

    inputs2 = {
        'cMin':10,
        'cMax':1800,
        'zMax':8,
        'cMap':'Spectral_r',
        'colorBar':True, 
        'orientation':'vertical',
        'label': '$\sigma$ [mS/m]',
        'fontsize':7, 
        'labelsize':7
    }
    
    
    fs=8
    fs2=10
    fig, ax = plt.subplots(1,3, figsize=(8,2), sharey=True, layout='constrained')
   
    showStitchedModels(model_GS, ax = ax[0], **inputs1)
    ax[0].plot(x,-z1,':k')
    ax[0].plot(x,-z1-z2,':k')
    ax[0].set_ylabel('Depth [m]', fontsize=fs)
    ax[0].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GS[:,2:]) + ' mS/m', fontsize=fs)
    ax[0].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GS[:,:2]) + ' m', fontsize=fs)
    ax[0].tick_params(labelsize=fs)
    ax[0].set_title('Estimated model - GS [Q+IP]', fontsize=fs2)
    ax[0].set_xlabel('Distance [m]', fontsize=fs)
    
    showStitchedModels(model_GN, ax = ax[1], **inputs1)
    #ax[1,1].set_ylabel('Depth [m]')
    ax[1].plot(x,-z1,':k')
    ax[1].plot(x,-z1-z2,':k')
    ax[1].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GN[:,2:]) + ' mS/m', fontsize=fs)
    ax[1].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GN[:,:2]) + ' m', fontsize=fs)  
    ax[1].tick_params(labelsize=fs)
    ax[1].set_title('Estimated model - GN [Q+IP]', fontsize=fs2)
    ax[1].set_xlabel('Distance [m]', fontsize=fs)

    showStitchedModels(model_GSGN, ax = ax[2], **inputs2)
    #ax[1,1].set_ylabel('Depth [m]')
    ax[2].plot(x,-z1,':k')
    ax[2].plot(x,-z1-z2,':k')
    ax[2].text(0,-6, 'RMSE $\sigma$: %2.3f' % root_mean_squared_error(model_true[:,2:], model_GSGN[:,2:]) + ' mS/m', fontsize=fs)
    ax[2].text(0,-7, 'RMSE $h$: %2.3f' % root_mean_squared_error(model_true[:,:2], model_GSGN[:,:2]) + ' m', fontsize=fs)  
    ax[2].tick_params(labelsize=fs)
    ax[2].set_title('Estimated model - GS + GN [Q+IP]', fontsize=fs2)
    ax[2].set_xlabel('Distance [m]', fontsize=fs)
    
def PlotModel(model, depths, ax=None, model_name=None, model_style='k', ylab=False, xlab=False, lw=1):
    """ Plot a 1D model """
    if ax is None:
        fig, ax = plt.subplots()
    fs=7
    ax.step(model, depths, color=model_style, label=model_name, linewidth=lw)
    ax.set_xscale('log')
    if xlab == True:
        ax.set_xlabel('Electrical conductivity [mS/m]', fontsize=fs)
    if ylab == True:
        ax.set_ylabel('Depth [m]', fontsize=fs)
    if model_name is not None:
        ax.legend(fontsize=fs, bbox_to_anchor=(1.1, 1.05))
    ax.tick_params(labelsize=fs)
    plt.tight_layout()

    
def Plot_m0(lambd, height, offsets, freq, filt, sigmas, models_m0, model_true, dmax=-10, ax=None, ylab=False, xlab=False, legend=False):
    """ Plot the results of GN estimation with different m0 """
    if ax is None:
        fig, ax = plt.subplots()
    fs=7
    
    for m in range(len(models_m0)):
        color = (0.1*m, 0.5, 0.5)
        mod = models_m0[m]
        sigma_2Lay_plot = np.hstack((mod[2:], mod[-1]))
        depths_2Lay_plot = np.hstack([0, -mod[0], -np.sum(mod[:2]), dmax])
        data = EMf_3Lay_HVP(lambd, mod[2], mod[3], mod[4], mod[0], mod[1],
                            height, offsets, freq, filt)
        data_true = EMf_3Lay_HVP(lambd, model_true[2], model_true[3], model_true[4], model_true[0], model_true[1],
                                 height, offsets, freq, filt)
        rmse = nrmse(data_true, data)
        if legend == True:
            PlotModel(sigma_2Lay_plot, depths_2Lay_plot, ax=ax, model_style=color, model_name='$m_0: $' 
                      + str(sigmas[m]) + ' mS/m' + ', nrmse: ' + '%.2f' %rmse)
        else:
            PlotModel(sigma_2Lay_plot, depths_2Lay_plot, ax=ax, model_style=color, )
    sigma_true = np.hstack((model_true[2:], model_true[-1]))
    depth_true = np.hstack([0, -model_true[0], -np.sum(model_true[:2]), dmax])
    if legend == True:
        PlotModel(sigma_true, depth_true, model_name='True', ax=ax, lw = 2)
    else:
        PlotModel(sigma_true, depth_true, ax=ax, lw=2)
        
def Plot_1DModel_3Lay(ax, model_true, model_GS, model_GN, model_ini, pos, 
                  case='', method='', depthmax=-8, colorbar=False):
    # Arrays to plot
    depth_true = np.array([0, -model_true[0], -np.sum(model_true[:2]), depthmax])
    depth_GS = np.array([0, -model_GS[0], -np.sum(model_GS[:2]), depthmax])
    depth_GN = np.array([0, -model_GN[0], -np.sum(model_GN[:2]), depthmax])
    depth_ini = np.array([0, -model_ini[0], -np.sum(model_ini[:2]), depthmax])

    sigma_true = np.hstack([model_true[2:], model_true[-1]])
    sigma_GS = np.hstack([model_GS[2:], model_GS[-1]])
    sigma_GN = np.hstack([model_GN[2:], model_GN[-1]])
    sigma_ini = np.hstack([model_ini[2:], model_ini[-1]])
    
    ax.step(sigma_true*1000, depth_true, 'k', label = 'True', linewidth=4)
    ax.step(sigma_GS*1000, depth_GS, 'r', label='GS')
    ax.step(sigma_GN*1000, depth_GN, 'c', label='GN')
    ax.step(sigma_ini*1000, depth_ini, 'g', label='Initial')
    ax.set_xlim([5,2500])
    ax.set_ylabel('Depth [m]', fontsize=8)
    ax.set_xlabel('$\sigma$ [mS/m]', fontsize=8)
    ax.set_title('1D Model X=' +str(pos) + 'm - Case: '+case, fontsize=8)
    ax.set_xscale('log')
    ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize=7)
    ax.tick_params(axis='both',labelsize=9)
    
def Plot_SolSpa_3Lay(models_err, err, model_true, model_GS, model_GN, model_ini, model_GN_hist, pos, title=''):
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    # Extract models from solution space where the lookup table model sigma_1 == estimated model sigma_1
    index_s1 = models_err[:,2] == model_GS[pos,2]  # obtain indices
    models_err = models_err[np.where(index_s1)[0]] # extract models from indices
    err = err[np.where(index_s1)[0]] # extract error from indices

    # Extract models from solution space where the lookup table model sigma_3 == estimated model sigma_3
    index_s3 = models_err[:,4] == model_GS[pos,4]
    models_err = models_err[np.where(index_s3)[0]]
    err = err[np.where(index_s3)[0]]

    # defining axes
    x = np.log10(models_err[:,3]*1000) # log10(conductivity) of the second layer
    y = models_err[:,0] # thickness of first layer
    z = models_err[:,1] # thickness of second layer
    c = err*100 # error values
    
    # Plot
    cb = ax.scatter(x, y, z, c = c, s= 0.01, cmap="RdBu_r")
    ax.scatter(np.log10(model_GS[pos,3]*1000), model_GS[pos,0], model_GS[pos,1], c='r', s=20, label='GS')
    ax.scatter(np.log10(model_true[pos,3]*1000), model_true[pos,0], model_true[pos,1], c='k', s=20, label='True')
    ax.scatter(np.log10(model_GN[pos,3]*1000), model_GN[pos,0], model_GN[pos,1], c='y', s=20, label='GN')
    ax.scatter(np.log10(model_ini[3]*1000), model_ini[0], model_ini[1], c='g', s=20, label='Initial')
    ax.legend(fontsize=7)
    
    # Plot optimization history
    for i in range(len(model_GN_hist)):
        if i >= 1:
            m1 = model_GN_hist[i-1]
            m2 = model_GN_hist[i]
            x = np.log10(model_GN_hist[i-1:i+1,3]*1000)
            y = model_GN_hist[i-1:i+1,0]
            z = model_GN_hist[i-1:i+1,1]
            ax.plot(x,y,z, ':k')
    ax.view_init(30, 30,0)
    ax.set_xlabel('$log_{10}$($\sigma_2$) [mS/m]')
    ax.set_ylabel('$h_1$ [m]')
    ax.set_zlabel('$h_2$ [m]')
    ax.set_title(title)

    clb = fig.colorbar(cb, ax=ax, ticks=[0, 5, 10, 15, 20, 25, 30], shrink=0.5)
    clb.ax.set_title('NRMSE %', fontsize=7)
    clb.ax.tick_params(labelsize=9)
    plt.show()
    plt.tight_layout()
    
def PlotLine(ax, model, elevation, dist, vmin=1, vmax=500, depthmax=8, colorbar=False, xlab=False, ylab=False):
    """ Function to plot the estimated positions in a stitched 2D section
    
    Parameters:
    1. model: Estimated model in each position from the global search 
       m = [sigma_1, sigma_2, h_1]
    2. elevation: Elevation values [m]
    3. dist = Horizontal distance of the section [m]
    4. vmin = minimum value for colorbar
    5. vmax = maximum value for colorbar
    
    """
    
    # Arrays for plotting
    
    npos = len(model)
    depthmax = -np.min(elevation)+depthmax
    
    ny = 101
    y = np.linspace(0, depthmax, ny)
    sigy = np.zeros((len(model), ny))
    
    sigma_1 = model[:,1]
    sigma_2 = model[:,2]
    thick_1 = model[:,0]
    
    depth = thick_1 -(elevation)
    
    # Conductivities array to be plotted
    for i in range(npos):
        y0 = 0
        while y[y0] <= -elevation[i]: # - sign because the elevations are negative!!
            sigy[i, y0:] = 0
            y0 += 1       
        while y[y0] <= depth[i]:
            sigy[i, y0:] = sigma_1[i]
            y0 += 1
        sigy[i, y0:] = sigma_2[i]
            
    #fig, ax = plt.subplots(figsize = (7,6))
    pos = ax.imshow((sigy*1000).T, cmap='Spectral_r', interpolation='none', 
                    extent= [0,dist,depthmax+2,0], vmin = vmin, vmax=vmax, norm='log' )

    if colorbar==True:
        clb = fig.colorbar(pos, shrink=0.4)
        #clb.ax.tick_params(labelsize=8)
        clb.set_label('$\sigma$ [mS/m]',  )

    if ylab == True:
        ax.set_ylabel('Depth [m]', fontsize=8)
    if xlab == True:
        ax.set_xlabel('Distance [m]', fontsize=8)

    ax.tick_params(labelsize=8)
    return pos
    

def distance(lat1, lat2, lon1, lon2):
    """ Function to calculate the horizontal distance between to 
    coordinates in degrees """
    # Approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(lat1)  
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c # distance in km
    return distance*1000 # distance in m