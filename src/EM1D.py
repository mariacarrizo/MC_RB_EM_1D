#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Name: EM1D.py
Description: This script contains the Forward Functions and Inversion classes
Author: Maria Carrizo
Email: m.e.carrizomascarell@tudelft.nl
Date created: 15/12/2023
"""

import numpy as np
from scipy.constants import mu_0
import pygimli as pg
from itertools import product
from sklearn.metrics import root_mean_squared_error

#%%

def nrmse(obs, pred):
    """ Normalized rmse 
    obs : observed
    pred : predicted
    """
    return root_mean_squared_error(obs, pred)/np.abs(np.max(obs)-np.min(obs))

# Functions to calculate mutual impedance ratios

def Z_H(s, R_0, lambd, a, filt):
    """ Function to calculate the mutual impedance for a horizontal
    coplanar (H) coil orientation in a 2-layered 1D model
    
    Input:
        s : coil separation in m
        R_0 : reflection coefficient
        lambd : radial component of the wavenumber
        a : air layer thickness in m
        filt : filter to perform the hankel transform
        
    Output:
        Z : mutual impedance for each offset
    
    """
    Z = - s**3 * (np.dot((R_0*np.exp(-2*lambd*a)*lambd**2), filt.j0)/s)
    return Z

def Z_V(s, R_0, lambd, a, filt):
    """ Function to calculate the mutual impedance for a vertical
    coplanar (V) coil orientation in a 2-layered 1D model
    
    Input:
        s : coil separation in m
        R_0 : reflection coefficient
        lambd : radial component of the wavenumber
        a : air layer thickness in m
        filt : filter to perform the hankel transform
        
    Output:
        Z : mutual impedance for each offset
    
    """
    Z = - s**2 * (np.dot((R_0*np.exp(-2*lambd*a)*lambd), filt.j1)/s)
    return Z

def Z_P(s, R_0, lambd, a, filt):
    """ Function to calculate the mutual impedance for a perpendicular
    (P) coil orientation in a 2-layered 1D model
    
    Input:
        s : coil separation in m
        R_0 : reflection coefficient
        lambd : radial component of the wavenumber
        a : air layer thickness in m
        filt : filter to perform the hankel transform
        
    Output:
        Z : mutual impedance for each offset
    
    """
    Z = s**3 * (np.dot((R_0*np.exp(-2*lambd*a)*lambd**2),filt.j1)/s)
    return Z

#%%
def R0_2Lay(lam, sigma1, sigma2, h1, freq):
    """ Recursive function to calculate reflection coefficient R_0 for a 2-layered model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        h1 : thickness of 1st layer in m
        freq : frequency in Hertz
        
    Output:
        r0 : reflection coefficient
    """
    gam1 = np.sqrt(lam**2 + 1j*(2 * np.pi * freq) *sigma1 * mu_0)
    gam2 = np.sqrt(lam**2 + 1j*(2 * np.pi * freq) *sigma2 * mu_0)
    r1 = (gam1 - gam2)/(gam1 + gam2)
    r0 = ((lam - gam1)/(lam + gam1) + r1 * np.exp(- 2 * gam1 * h1))/(1 + (lam - gam1)/(lam
           + gam1) * r1 * np.exp(-2 * gam1 * h1))
    return r0

def R0_3Lay(lam, sigma1, sigma2, sigma3, h1, h2, freq):
    """ Recursive function to calculate reflection coefficient R_0 for a 3-layered model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        sigma3 : electrical conductivity of 3rd layer in S/m
        h1 : thickness of 1st layer in m
        h2 : thickness of 2nd layer in m
        freq : frequency in Hertz
        
    Output:
        r0 : reflection coefficient
    """
    gam1 = np.sqrt(lam**2 + 1j*(2 * np.pi *freq) *sigma1 * mu_0)
    gam2 = np.sqrt(lam**2 + 1j*(2 * np.pi *freq) *sigma2 * mu_0)
    gam3 = np.sqrt(lam**2 + 1j*(2 * np.pi *freq) *sigma3 * mu_0)
    r2 = (gam2 - gam3)/(gam2 + gam3)
    r1 = ((gam1 - gam2)/(gam1 + gam2) + r2 * np.exp(-2 * gam2 * h2))/(1 + (gam1 - gam2)/(gam1 
            + gam2) * r2 * np.exp(-2 * gam2 * h2))
    r0 = ((lam - gam1)/(lam + gam1) + r1 * np.exp(-2 * gam1 * h1))/(1 + (lam - gam1)/(lam 
            + gam1) * r1 * np.exp(-2 * gam1 * h1))
    return r0 

# Forward function

def EMf_2Lay_HVP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt):
    """ Forward function for a 2-layered earth model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        h1 : thickness of 1st layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : vector with the quadrature Q and in-phase IP components of the 
               measurements for the H, V and P coil orientations as
              [Q_H, Q_V, Q_P, IP_H, IP_V, IP_P]
    """
    # Calculate reflection coefficient
    R0 = R0_2Lay(lambd, sigma1, sigma2, h1, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets+0.1, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_v = np.abs(Z_v.imag)
    Q_p = np.abs(Z_p.imag)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real)
    IP_v = np.abs(Z_v.real)
    IP_p = np.abs(Z_p.real)
    
    return np.hstack((Q_h, Q_v, Q_p, IP_h, IP_v, IP_p))

def EMf_2Lay_HVP_Q(lambd, sigma1, sigma2, h1, height, offsets, freq, filt):
    """ Forward function for a 2-layered earth model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        h1 : thickness of 1st layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : vector with the quadrature Q component of the 
               measurements for the H, V and P coil orientations as
              [Q_H, Q_V, Q_P]
    """
    # Calculate reflection coefficient
    R0 = R0_2Lay(lambd, sigma1, sigma2, h1, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets+0.1, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_v = np.abs(Z_v.imag)
    Q_p = np.abs(Z_p.imag)
    
    return np.hstack((Q_h, Q_v, Q_p))

def EMf_2Lay_HVP_IP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt):
    """ Forward function for a 2-layered earth model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        h1 : thickness of 1st layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : vector with the in-phase IP components of the 
               measurements for the H, V and P coil orientations as
              [IP_H, IP_V, IP_P]
    """
    # Calculate reflection coefficient
    R0 = R0_2Lay(lambd, sigma1, sigma2, h1, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets+0.1, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real)
    IP_v = np.abs(Z_v.real)
    IP_p = np.abs(Z_p.real)
    
    return np.hstack((IP_h, IP_v, IP_p))
    
def GlobalSearch_2Lay(Database, Data, conds, thicks, nsl=51):
    """ This function searches through the lookup table database
    for the best data fit, and then finds the corresponding model

    Parameters:
    1. Database: Lookup table
    2. Data: measurement for one position
    3. conds: Conductivities sampled in the lookup table
    4. thicks: thicknesses sampled in the lookup table
    5. nsl: number of samples

    Returns: 2 layered model estimated through best data fit
    model = [sigma_1, sigma_2, h_1]

    Units:
    sigma_1 [S/m]
    sigma_2 [S/m]
    h_1 [m]
    """

    err = 1
    indx = 0

    # Search best data fit
    for i in range(np.shape(Database)[0]):
        nZdiff = ((Database[i] - Data) **2) / (Data)**2
        merr = np.log10(np.sqrt(np.sum(nZdiff)/len(Data)))
        if merr < err:
            indx = i
            err = merr.copy()

    # Find corresponding model
    for i in range(len(conds)):
        for j in range(len(conds)):
            for k in range(len(thicks)):
                idx = k + j*nsl + i*nsl**2
                if indx == idx:
                    model = np.array([thicks[k], conds[i], conds[j]])

    return model

class EMf_2Lay_GN_HVP_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the quadrature (Q) and in-phase (IP) components of the measurements
    for a 2-layered model
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """   
    def __init__(self, lambd, height, offsets, freq, filt, nlay=2):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_2Lay_HVP(lambd = self.lambd,
                            sigma1 = np.asarray(par)[1], 
                            sigma2 = np.asarray(par)[2], 
                            h1 = np.asarray(par)[0], 
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')
        
class EMf_3Lay_GN_HVP_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the quadrature (Q) and in-phase (IP) components of the measurements
    for a 3-layered model
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """
    def __init__(self, lambd, height, offsets, freq, filt, nlay=3):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_3Lay_HVP(lambd = self.lambd,
                            sigma1 = np.asarray(par)[2], 
                            sigma2 = np.asarray(par)[3],
                            sigma3 = np.asarray(par)[4],
                            h1 = np.asarray(par)[0], 
                            h2 = np.asarray(par)[1],
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')

class EMf_3Lay_GN_HV_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the quadrature (Q) and in-phase (IP) components of the measurements
    for a 3-layered model, using only H and V coil geometries
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """
    def __init__(self, lambd, height, offsets, freq, filt, nlay):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_3Lay_HV_field(lambd = self.lambd,
                            sigma1 = np.asarray(par)[2], 
                            sigma2 = np.asarray(par)[3],
                            sigma3 = np.asarray(par)[4],
                            h1 = np.asarray(par)[0], 
                            h2 = np.asarray(par)[1],
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')
        
class EMf_2Lay_GN_HVP_Q_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the quadrature (Q) component of the measurements
    for a 2-layered model
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """
    def __init__(self, lambd, height, offsets, freq, filt, nlay=2):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_2Lay_HVP_Q(lambd = self.lambd,
                            sigma1 = np.asarray(par)[1], 
                            sigma2 = np.asarray(par)[2], 
                            h1 = np.asarray(par)[0], 
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')
        
class EMf_3Lay_GN_HVP_Q_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the quadrature (Q) component of the measurements
    for a 3-layered model
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """
    def __init__(self, lambd, height, offsets, freq, filt, nlay=3):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_3Lay_HVP_Q(lambd = self.lambd,
                            sigma1 = np.asarray(par)[2], 
                            sigma2 = np.asarray(par)[3], 
                            sigma3 = np.asarray(par)[4],
                            h1 = np.asarray(par)[0], 
                            h2 = np.asarray(par)[1],
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')
        
class EMf_2Lay_GN_HVP_IP_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the in-phase (IP) component of the measurements
    for a 2-layered model
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """
    def __init__(self, lambd, height, offsets, freq, filt, nlay=2):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
            
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_2Lay_HVP_IP(lambd = self.lambd,
                            sigma1 = np.asarray(par)[1], 
                            sigma2 = np.asarray(par)[2], 
                            h1 = np.asarray(par)[0], 
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')
        
class EMf_3Lay_GN_HVP_IP_1D(pg.frameworks.Modelling):
    """ Class to Initialize the model for Gauss-Newton inversion
    using the in-phase (IP) component of the measurements
    for a 3-layered model
    
    Input:
        lambd : radial component of the wavenumber
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        nlay : number of layers
    """
    def __init__(self, lambd, height, offsets, freq, filt, nlay=3):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
            
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_3Lay_HVP_IP(lambd = self.lambd,
                            sigma1 = np.asarray(par)[2], 
                            sigma2 = np.asarray(par)[3],
                            sigma3 = np.asarray(par)[4],
                            h1 = np.asarray(par)[0], 
                            h2 = np.asarray(par)[1],
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')

def ErrorSpace_2Lay(Database, Data, max_error, conds, thicks, nsl=51):
    """ Returns the models and relative error of the models in the lookup table 
    below a max error
    
    Input:
        Database : Lookup table
        Data : data array for a single 1D model
        max_error : error maximum boundary for the error space map
        conds : electrical conductivities sampled in the Database [S/m]
        thicks : thicknesses sampled in the Database [m] 
        
    Output:
        err : array with the error values for the models with data 
              misfit < max_error
        models_below_err : models with data misfit < max_error
    """
    err = []
    models_below_err = []
    for d in range(np.shape(Database)[0]):       
        diff = np.abs((Database[d] - Data)/Data)
        merr = np.sum(diff)/len(Data)
        
        if merr < max_error:
            err.append(merr)
            indx = d
            for i in range(len(conds)):
                for j in range(len(conds)):
                    for k in range(len(thicks)):
                        idx = k + j*nsl + i*nsl**2
                        if indx == idx:
                            model = np.array([conds[i], conds[j], thicks[k]])
                            models_below_err.append(model)
    return np.array(err), np.array(models_below_err)

def EMf_3Lay_HVP(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt):
    """ Forward function for a 3-layered earth model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        sigma3 : electrical conductivity of 3rd layer in S/m
        h1 : thickness of 1st layer in m
        h2 : thickness of 2nd layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : vector with the Quadrature (Q) and In-phase (IP) components of the 
               measurements for the H, V and P coil orientations as
              [Q_H, Q_V, Q_P, IP_H, IP_V, IP_P]
              
    """
    # Calculate reflection coefficient
    R0 = R0_3Lay(lambd, sigma1, sigma2, sigma3, h1, h2, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets+0.1, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_v = np.abs(Z_v.imag)
    Q_p = np.abs(Z_p.imag)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real)
    IP_v = np.abs(Z_v.real)
    IP_p = np.abs(Z_p.real)
    
    return np.hstack((Q_h, Q_v, Q_p, IP_h, IP_v, IP_p))

def EMf_3Lay_HV_field(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt):
    """ Forward function for a 3-layered earth model
        Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        sigma3 : electrical conductivity of 3rd layer in S/m
        h1 : thickness of 1st layer in m
        h2 : thickness of 2nd layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        Output:
        data : vector with the Quadrature (Q) and In-phase (IP) components of the
        measurements for the H, V and P coil orientations as
        [Q_H, Q_V, IP_H, IP_V]
    """
    # Calculate reflection coefficient
    R0 = R0_3Lay(lambd, sigma1, sigma2, sigma3, h1, h2, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_v = np.abs(Z_v.imag)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real[1:])
    IP_v = np.abs(Z_v.real[1:])
    return np.hstack((Q_h, Q_v, IP_h, IP_v))

def EMf_3Lay_HP_field(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt):
    """ Forward function for a 3-layered earth model
        Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        sigma3 : electrical conductivity of 3rd layer in S/m
        h1 : thickness of 1st layer in m
        h2 : thickness of 2nd layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        Output:
        data : vector with the Quadrature (Q) and In-phase (IP) components of the
        measurements for the H, V and P coil orientations as
        [Q_H, IP_H, Q_P, IP_P]
    """
    # Calculate reflection coefficient
    R0 = R0_3Lay(lambd, sigma1, sigma2, sigma3, h1, h2, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_p = np.abs(Z_p.imag)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real)
    IP_p = np.abs(Z_p.real)
    return np.hstack((Q_h, IP_h, Q_p, IP_p))

def EMf_2Lay_HP_field(lambd, sigma1, sigma2, h1, height, offsets, freq, filt):
    """ Forward function for a 3-layered earth model
        Input:
            lam : radial component of the wavenumber
            sigma1 : electrical conductivity of 1st layer in S/m
            sigma2 : electrical conductivity of 2nd layer in S/m
            h1 : thickness of 1st layer in m
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
        Output:
            data : vector with the Quadrature (Q) and In-phase (IP) components of the
                   measurements for the H and P coil orientations as
                   [Q_H, IP_H, Q_P, IP_P]
    """
    # Calculate reflection coefficient
    R0 = R0_2Lay(lambd, sigma1, sigma2, h1, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_p = np.abs(Z_p.imag)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real)
    IP_p = np.abs(Z_p.real)
    return np.hstack((Q_h, IP_h, Q_p, IP_p))

def EMf_3Lay_HVP_Q(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt):
    """ Forward function for a 2-layered earth model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        sigma3 : electrical conductivity of 3rd layer in S/m
        h1 : thickness of 1st layer in m
        h2 : thickness of 2nd layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : vector with the Quadrature (Q) component of the 
               measurements for the H, V and P coil orientations as
              [Q_H, Q_V, Q_P]
              
    """
    # Calculate reflection coefficient
    R0 = R0_3Lay(lambd, sigma1, sigma2, sigma3, h1, h2, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets+0.1, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_v = np.abs(Z_v.imag)
    Q_p = np.abs(Z_p.imag)
    
    return np.hstack((Q_h, Q_v, Q_p))

def EMf_3Lay_HVP_IP(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt):
    """ Forward function for a 2-layered earth model
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        sigma3 : electrical conductivity of 3rd layer in S/m
        h1 : thickness of 1st layer in m
        h2 : thickness of 2nd layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : vector with the In-phase (IP) component of the 
               measurements for the H, V and P coil orientations as
              [IP_H, IP_V, IP_P]
              
    """
    # Calculate reflection coefficient
    R0 = R0_3Lay(lambd, sigma1, sigma2, sigma3, h1, h2, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets+0.1, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain in-phases
    IP_h = np.abs(Z_h.real)
    IP_v = np.abs(Z_v.real)
    IP_p = np.abs(Z_p.real)
    
    return np.hstack((IP_h, IP_v, IP_p))

def GlobalSearch_3Lay(Database, Data, conds, thicks, nsl=51):
    """ This function searches through the lookup table database
    for the best data fit, and then finds the corresponding model

    Parameters:
    1. Database: Lookup table
    2. Data: measurement for one position
    3. conds: Conductivities sampled in the lookup table in S/m
    4. thicks: thicknesses sampled in the lookup table in m
    5. nsl: number of samples

    Returns: 3 layered model estimated through best data fit
    model = [h_1, h_2, sigma_1, sigma_2, sigma_3]

    Units:
    sigma_1 [S/m]
    sigma_2 [S/m]
    sigma_3 [S/m]
    h_1 [m]
    h_2 [m]
    """
    
    # Evaluate for min error
    nZdiff = ((Database[:] - Data)**2)/(Data**2)
    rmse_vector = np.sqrt(np.sum(nZdiff, axis=1)/len(Data))
    indx_min_rmse = np.argmin(rmse_vector)

    # Return model that corresponds to the index
    ## using itertool product we create a small array Indices
    Indices = np.array(list(product(range(nsl), range(nsl), range(nsl), 
                                    range(nsl), range(nsl))),dtype=int)
    
    # Index of each model parameter for the min rmse
    m_idx = Indices[indx_min_rmse]
    
    # Estimated model from global search
    model = np.array([thicks[m_idx[3]], thicks[m_idx[4]], conds[m_idx[0]], conds[m_idx[1]], conds[m_idx[2]]])
    
    return model
  
def EMf_2Lay_HV_field(lambd, sigma1, sigma2, h1, height, offsets, freq, filt):
    """ Forward function for a 2-layered earth model in field case
    
    Input:
        lam : radial component of the wavenumber
        sigma1 : electrical conductivity of 1st layer in S/m
        sigma2 : electrical conductivity of 2nd layer in S/m
        h1 : thickness of 1st layer in m
        height : height of the instrument above ground in m
        offsets : coil separation in m
        freq : frequency in Hertz
        filt : filter to perform the hankel transform
        
    Output:
        data : array with the quadrature (Q) and in-phase (IP) components of the 
               measurements for the H, V coil orientations as. 
               In-phase components for offsets > 4m
              [Q_H, Q_V, IP_H, IP_V]
    
    """
    # Calculate reflection coefficient
    R0 = R0_2Lay(lambd, sigma1, sigma2, h1, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = np.abs(Z_h.imag)
    Q_v = np.abs(Z_v.imag)
    # Obtain in-phases
    IP_h = np.abs((Z_h.real)[1:]) # only for the offsets >2
    IP_v = np.abs((Z_v.real)[1:]) # only for the offsets >2
    return np.hstack((Q_h, Q_v, IP_h, IP_v))


class EMf_2Lay_GN_HV_field(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt, nlay):
        self.nlay = nlay
        mesh = pg.meshtools.createMesh1DBlock(nlay)
        super().__init__()
        self.setMesh(mesh)
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
            
    def response(self, par):
        """ Compute response vector for a certain model [mod] 
        par = [thickness_1, thickness_2, ..., thickness_n, sigma_1, sigma_2, ..., sigma_n]
        """
        resp = EMf_2Lay_HV_field(lambd = self.lambd,
                            sigma1 = np.asarray(par)[1], 
                            sigma2 = np.asarray(par)[2], 
                            h1 = np.asarray(par)[0], 
                            height = self.height, 
                            offsets = self.offsets, 
                            freq = self.freq, 
                            filt = self.filt
                              )
        return resp
    
    def response_mt(self, par, i=0):
        """Multi-threaded forward response."""
        return self.response(par)
    
    def createJacobian(self, par, dx=1e-4):
        """ compute Jacobian for a 1D model """
        resp = self.response(par)
        n_rows = len(resp) # number of data values in data vector
        n_cols = len(par) # number of model parameters
        J = self.jacobian() # we define first this as the jacobian
        J.resize(n_rows, n_cols)
        Jt = np.zeros((n_cols, n_rows))
        for j in range(n_cols):
            mod_plus_dx = par.copy()
            mod_plus_dx[j] += dx
            Jt[j,:] = (self.response(mod_plus_dx) - resp)/dx # J.T in col j
        for i in range(n_rows):
            J[i] = Jt[:,i]
        #print(self.jacobian())
        #print(J)
        #print(Jt)
        
    def drawModel(self, ax, model):
        pg.viewer.mpl.drawModel1D(ax = ax,
                                  model = model,
                                  plot = 'semilogx',
                                  xlabel = 'Electrical conductivity (S/m)',
                                  )
        ax.set_ylabel('Depth in (m)')
    

    
def Q_from_Sigma(sigma, s, freq=9000, mu_0=mu_0):
    """ Function that back transforms Sigma_app to Quadrature values
    using the LIN approximation function 
    
    Parameters: 
    1. sigma: apparent conductivity [S/m]
    2. s: coil offset [m]
    
    Returns:
    Q : quadrature values
    """
    
    Q = sigma * (2 *np.pi * freq) * mu_0 * s**2 /4
    return Q

# Global search noise analysis
def NoiseAnalysis_GS_2Lay(LUT, conds, thicks, nsl, data_true, noise=0.1, test_it=100):
    """ 
    Function to estimate a model using global search for a percentage of noise
    in a number of random instances
    LUT : Lookup table
    conds : conductivities sampled in the lookup table
    thicks : thicknesses sampled in the lookup table
    nsl : sampling number of the lookup table
    data_true : true data
    noise : noise percentage
    tes_it : number of instances
    """
    
    models_GS = []
    for i in range(test_it):
        data_noise = data_true* (1 +np.random.normal(size=len(data_true))*noise)
        models_GS.append(GlobalSearch_2Lay(LUT, data_noise, conds, thicks, nsl))
    return models_GS 

# Gauss-Newton noise analysis
def NoiseAnalysis_GN_2Lay(lambd, height, offsets, freq, filt, model_ini, data_true, noise=0.1, test_it=100):
    """ 
    Function to estimate a model using Gauss-Newton for a percentage of noise
    in a number of random instances
    data_true : true data
    noise : noise percentage
    tes_it : number of instances
    """
    m0 = model_ini
    lam = 0
    models_GN = []

    transThk = pg.trans.TransLogLU(0.1,7)
    transSig = pg.trans.TransLogLU(10/1000,2000/1000)
    error = 1e-3 # relative error
    relativeError = np.ones_like(data_true[0]) * error # Relative error array
    
    for i in range(test_it):
        # Define forward modelling class
        EMf = EMf_2Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=2)

        # Define transformation
        EMf.region(0).setTransModel(transThk)
        EMf.region(1).setTransModel(transSig)

        # Define inversion framework from pygimli
        invEM = pg.Inversion()
        invEM.setForwardOperator(EMf) # set forward operator

        data_noise = data_true* (1 +np.random.normal(size=len(data_true))*noise)
        models_GN.append(invEM.run(data_noise, relativeError, startModel= m0, lam=lam, verbose=False))
    return models_GN

def SolSpa_2Lay_sigma1(lambd, height, offsets, freq, filt, model_true, data_true, pos, thicks, conds, max_nrmse=0.4):
    """ Function to evaluate the solution space in a 2-Layered model
    for a fixed value of sigma_1
    model_true : true model
    data_true : true data
    pos : position of the 1D model to evaluate
    thicks : thicknesses sampled in the solution space
    conds : conductivities sampled in the solution space 
    max_nrmse : a maximum value of the nrmse to evaluate in the solution space
    
    returns error values and models sampled in the solution space
    """
    
    err = [] # to store error values
    models_err = [] # to store the models of the solution space
    
    # evaluate solution space
    for h1 in thicks:
        for sigma2 in conds:
            sigma1 = model_true[pos,1]
            mod = [h1, sigma1, sigma2]
            dat = EMf_2Lay_HVP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
            n = nrmse(data_true[pos], dat) # calculate normalized rmse for the sampled model
            
            # if the error is below max_err, append
            if n < max_nrmse:
                err.append(n)    
                models_err.append(mod)

    # convert into numpy arrays
    err = np.array(err)
    models_err = np.array(models_err)

    return err, models_err

def SolSpa_2Lay_sigma2(lambd, height, offsets, freq, filt, model_true, data_true, pos, thicks, conds, max_nrmse=0.4):
    """ Function to evaluate the solution space in a 2-Layered model
    for a fixed value of sigma_2
    model_true : true model
    data_true : true data
    pos : position of the 1D model to evaluate
    thicks : thicknesses sampled in the solution space
    conds : conductivities sampled in the solution space 
    max_nrmse : a maximum value of the nrmse to evaluate in the solution space
    
    returns error values and models sampled in the solution space
    """
        
    err = [] # to store error values
    models_err = [] # to store the models of the solution space
    
    # evaluate solution space
    for h1 in thicks:
        for sigma1 in conds:
            sigma2 = model_true[pos,2]
            mod = [h1, sigma1, sigma2]
            dat = EMf_2Lay_HVP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
            n = nrmse(data_true[pos], dat) # calculate normalized rmse for the sampled model
            
            # if the error is below max_err, append
            if n < max_nrmse:
                err.append(n)    
                models_err.append(mod)

    # convert into numpy arrays
    err = np.array(err)
    models_err = np.array(models_err)

    return err, models_err

def SolSpa_2Lay_sigma1_Q(lambd, height, offsets, freq, filt, model_true, data_true, pos, thicks, conds, max_nrmse=0.4):
    """ Function to evaluate the solution space in a 2-Layered model
    for a fixed value of sigma_1, using only Q data
    
    model_true : true model
    data_true : true data
    pos : position of the 1D model to evaluate
    thicks : thicknesses sampled in the solution space
    conds : conductivities sampled in the solution space 
    max_nrmse : a maximum value of the nrmse to evaluate in the solution space
    
    returns error values and models sampled in the solution space
    """
    
    err = [] # to store error values
    models_err = [] # to store the models of the solution space
    
    # evaluate solution space
    for h1 in thicks:
        for sigma2 in conds:
            sigma1 = model_true[pos,1]
            mod = [h1, sigma1, sigma2]
            dat = EMf_2Lay_HVP_Q(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
            n = nrmse(data_true[pos,:9], dat)
            
            # if the error is below max_err, append
            if n < max_nrmse:
                err.append(n)    
                models_err.append(mod)

    # convert into numpy arrays
    err = np.array(err)
    models_err = np.array(models_err)

    return err, models_err

def SolSpa_2Lay_sigma2_Q(lambd, height, offsets, freq, filt, model_true, data_true, pos, thicks, conds, max_nrmse=0.4):
    """ Function to evaluate the solution space in a 2-Layered model
    for a fixed value of sigma_2 using only Q data
    
    model_true : true model
    data_true : true data
    pos : position of the 1D model to evaluate
    thicks : thicknesses sampled in the solution space
    conds : conductivities sampled in the solution space 
    max_nrmse : a maximum value of the nrmse to evaluate in the solution space
    
    returns error values and models sampled in the solution space
    """
        
    err = [] # to store error values
    models_err = [] # to store the models of the solution space
    
    # evaluate solution space
    for h1 in thicks:
        for sigma1 in conds:
            sigma2 = model_true[pos,2]
            mod = [h1, sigma1, sigma2]
            dat = EMf_2Lay_HVP_Q(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
            n = nrmse(data_true[pos,:9], dat) # calculate normalized rmse for the sampled model
            
            # if the error is below max_err, append
            if n < max_nrmse:
                err.append(n)    
                models_err.append(mod)

    # convert into numpy arrays
    err = np.array(err)
    models_err = np.array(models_err)

    return err, models_err

def SolSpa_2Lay_sigma1_IP(lambd, height, offsets, freq, filt, model_true, data_true, pos, thicks, conds, max_nrmse=0.4):
    """ Function to evaluate the solution space in a 2-Layered model
    for a fixed value of sigma_1, using only IP data
    
    model_true : true model
    data_true : true data
    pos : position of the 1D model to evaluate
    thicks : thicknesses sampled in the solution space
    conds : conductivities sampled in the solution space 
    max_nrmse : a maximum value of the nrmse to evaluate in the solution space
    
    returns error values and models sampled in the solution space
    """
    
    err = [] # to store error values
    models_err = [] # to store the models of the solution space
    
    # evaluate solution space
    for h1 in thicks:
        for sigma2 in conds:
            sigma1 = model_true[pos,1]
            mod = [h1, sigma1, sigma2]
            dat = EMf_2Lay_HVP_IP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
            n = nrmse(data_true[pos,9:], dat)
            
            # if the error is below max_err, append
            if n < max_nrmse:
                err.append(n)    
                models_err.append(mod)

    # convert into numpy arrays
    err = np.array(err)
    models_err = np.array(models_err)

    return err, models_err

def SolSpa_2Lay_sigma2_IP(lambd, height, offsets, freq, filt, model_true, data_true, pos, thicks, conds, max_nrmse=0.4):
    """ Function to evaluate the solution space in a 2-Layered model
    for a fixed value of sigma_2 using only IP data
    
    model_true : true model
    data_true : true data
    pos : position of the 1D model to evaluate
    thicks : thicknesses sampled in the solution space
    conds : conductivities sampled in the solution space 
    max_nrmse : a maximum value of the nrmse to evaluate in the solution space
    
    returns error values and models sampled in the solution space
    """
        
    err = [] # to store error values
    models_err = [] # to store the models of the solution space
    
    # evaluate solution space
    for h1 in thicks:
        for sigma1 in conds:
            sigma2 = model_true[pos,2]
            mod = [h1, sigma1, sigma2]
            dat = EMf_2Lay_HVP_IP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
            n = nrmse(data_true[pos,9:], dat) # calculate normalized rmse for the sampled model
            
            # if the error is below max_err, append
            if n < max_nrmse:
                err.append(n)    
                models_err.append(mod)

    # convert into numpy arrays
    err = np.array(err)
    models_err = np.array(models_err)

    return err, models_err

def SolSpa_3Lay_parallel(lambd, offsets, height, freq, filt, data_true, max_err, h1, h2, s1, s2, s3):
    """ Calculate solution space in a 3-layered case 
    
    data true : true data
    max_err : maximum error to evaluate
    h1 : thickness of first layer [m]
    h2 : thickness of second layer [m]
    s1 : conductivity of first layer [mS/m]
    s2 : conductivity of second layer [mS/m]
    s3 : conductivity of third layer [mS/m]
    """

    mod = [h1, h2, s1, s2, s3]
    dat = EMf_3Lay_HVP(lambd, s1, s2, s3, h1, h2, height, offsets, freq, filt)
    nrse = nrmse(data_true, dat)

    if nrse < max_err:
        model_err = np.hstack((mod, nrse))
        return model_err
    
def m0_Analysis_GN_3Lay(lambd, height, offsets, freq, filt, thicks, conds, sigma, data_true):
    """ Function to perform Gauss-Newton inversion with an specific initial model m0
    sigma : electrical conductivity of initial model
    """
    
    m0 = [3, 3, sigma/1000, sigma/1000, sigma/1000]
    lam = 0

    
    transThk = pg.trans.TransLogLU(np.min(thicks),np.max(thicks))
    transSig = pg.trans.TransLogLU(np.min(conds),np.max(conds))
    
    EMf = EMf_3Lay_GN_HVP_1D(lambd, height, offsets, freq, filt, nlay=3)

    # Define transformation
    EMf.region(0).setTransModel(transThk)
    EMf.region(1).setTransModel(transSig)

    # Define inversion framework from pygimli
    invEM = pg.Inversion()
    invEM.setForwardOperator(EMf) # set forward operator

    # Relative error array
    error = 1e-3 # relative error
    relativeError = np.ones_like(data_true[0]) * error
    model_GN = invEM.run(data_true, relativeError, startModel= m0, lam=lam, verbose=False)
    return model_GN
