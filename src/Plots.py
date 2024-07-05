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
def Plot_SolSpa_sigma1(ax1, ax2, model, model_GS, model_GN, model_ini, model_hist,
                       err, models_err, xmin=100, xmax=2000, ymin=0, ymax=7, case='', title='Solution Space', 
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
    ax1.step(sigma_Opt, depth_Opt, 'c', label='GI')
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
    ax2.scatter(model_Opt[2]*1000, model_Opt[0], marker ='.', c='y', label='GN', s=100)
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
    ax2.set_title(method, fontsize=8)
    ax2.set_xscale('log')
    
    if colorbar==True:
        clb = fig.colorbar(cntr1, ax=ax2, ticks=[0, 10, 20, 30, 40])
        clb.ax.set_title('NRMSE %', fontsize=7)
        clb.ax.tick_params(labelsize=9)
    
# Function to plot Solution space for a fixed sigma2
def Plot_SolSpa_sigma2(ax1, ax2, model, model_GS, model_GN, model_ini, model_hist,
                       err, models_err, xmin=10, xmax=600, ymin=0, ymax=7, case='', title='Solution Space', 
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
    ax1.step(sigma_Opt, depth_Opt, 'c', label='GI')
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
    ax2.scatter(model_Opt[1]*1000, model_Opt[0], marker ='.', c='y', label='GN', s=100)
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
    ax2.set_title(method, fontsize=8)
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
