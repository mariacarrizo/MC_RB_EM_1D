import numpy as np
from scipy.constants import mu_0
import pygimli as pg
from itertools import product

#%%
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
                    model = np.array([conds[i], conds[j], thicks[k]])

    return model

class EMf_2Lay_Opt_HVP(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
            using Quadrature and In-Phase components of the measurements
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
            
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of 2nd layer in S/m
        h1 = m[2]     # thickness of 1st layer in m
        Z = EMf_2Lay_HVP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)                           
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [3] # m
        sig_ini =  [100/1000, 100/1000] # S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)
    
class EMf_2Lay_Opt_HVP_Q(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
            using Quadrature component of the measurements
            
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
            
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        h1 = m[2] # thickness of 1st layer in m
        Z = EMf_2Lay_HVP_Q(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)                           
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [3] # m
        sig_ini =  [100/1000, 100/1000] # S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)
    
class EMf_2Lay_Opt_HVP_IP(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
            using In-Phase component of the measurements
            
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        h1 = m[2] # thickness of the 1st layer in m
        Z = EMf_2Lay_HVP_IP(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)                           
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [3] # m 
        sig_ini =  [100/1000, 100/1000] # in S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)

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

def GlobalSearch_3Lay(Database, Data, conds, thicks, norm, nsl=51):
    """ This function searches through the lookup table database
    for the best data fit, and then finds the corresponding model

    Parameters:
    1. Database: Lookup table
    2. Data: measurement for one position
    3. conds: Conductivities sampled in the lookup table in S/m
    4. thicks: thicknesses sampled in the lookup table in m
    5. nsl: number of samples

    Returns: 3 layered model estimated through best data fit
    model = [sigma_1, sigma_2, sigma_3, h_1, h_2]

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
    model = np.array([conds[m_idx[0]], conds[m_idx[1]], conds[m_idx[2]], 
                      thicks[m_idx[3]], thicks[m_idx[4]]])
    
    return model

class EMf_3Lay_Opt_HVP(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
        using the quadrature (Q) and in-phase (IP) components of the measurements
        
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        sigma3 = m[2] # electrical conductivity of the 3rd layer in S/m
        h1 = m[3] # thickness of 1st layer in m
        h2 = m[4] # thickness of 2nd layer in m
        # Perform forward function
        Z = EMf_3Lay_HVP(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt)
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [2,2] # Thicknesses in m
        sig_ini =  [50/1000, 50/1000, 50/1000] # conductivities in S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)
    
class EMf_3Lay_Opt_HVP_Q(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
        using the quadrature (Q) component of the measurements
        
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
        
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        sigma3 = m[2] # electrical conductivity of the 3rd layer in S/m
        h1 = m[3] # thickness of 1st layer in m
        h2 = m[4] # thickness of 2nd layer in m
        # Perform forward function
        Z = EMf_3Lay_HVP_Q(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt)
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [2,2] # Thicknesses in m
        sig_ini =  [50/1000, 50/1000, 50/1000] # conductivities in S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)
    
class EMf_3Lay_Opt_HVP_IP(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
        using the In-Phase (IP) component of the measurements
        
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
        
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        sigma3 = m[2] # electrical conductivity of the 3rd layer in S/m
        h1 = m[3] # thickness of 1st layer in m
        h2 = m[4] # thickness of 2nd layer in m
        # Perform forward function
        Z = EMf_3Lay_HVP_IP(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt)
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [2,2] # Thicknesses in m
        sig_ini =  [50/1000, 50/1000, 50/1000] # conductivities in S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)
    
class EMf_3Lay_GSplusOpt_HVP(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt, m0):
        """ Class to Initialize the model for the combined algorithm of global
        search + gradient based inversion 
        
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
            m0 : initial model from global search
        
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
        self.m0 = m0 # initial model from global search
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        sigma3 = m[2] # electrical conductivity of the 3rd layer in S/m
        h1 = m[3] # thickness of 1st layer in m
        h2 = m[4] # thickness of 2nd layer in m
        # Perform forward function
        Z = EMf_3Lay_HVP(lambd, sigma1, sigma2, sigma3, h1, h2, height, offsets, freq, filt)
        return Z               
    def createStartModel(self, dataVals):
        m0 = self.m0
        return m0
    
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


class EMf_2Lay_Opt_HV_field(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt):
        """ Class to Initialize the model for Gradient descent inversion
        for the field case 2-layered models
        
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
            
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        h1 = m[2] # thickness of the 1st layer
        # Perform the forward function
        Z = EMf_2Lay_HV_field(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)                           
        return Z               
    def createStartModel(self, dataVals):
        thk_ini = [2] # m
        sig_ini =  [100/1000, 100/1000]  # S/m
        m0 = sig_ini + thk_ini
        return np.array(m0)
    
class EMf_2Lay_GSplusOpt_HV_field(pg.Modelling):
    def __init__(self, lambd, height, offsets, freq, filt, m0):
        """ Class to Initialize the model for the combined algorithm of global
        search plus gradient descent inversion for the field case 2-layered 
        models
        
        Input:
            lambd : radial component of the wavenumber
            height : height of the instrument above ground in m
            offsets : coil separation in m
            freq : frequency in Hertz
            filt : filter to perform the hankel transform
            m0 : initial model coming from the global search
            
        """
        super().__init__()        
        self.lambd = lambd
        self.height = height
        self.offsets = offsets
        self.freq = freq
        self.filt = filt
        self.m0 = m0
    def response(self, m):
        lambd = self.lambd
        height = self.height
        offsets = self.offsets
        freq = self.freq
        filt = self.filt
        sigma1 = m[0] # electrical conductivity of the 1st layer in S/m
        sigma2 = m[1] # electrical conductivity of the 2nd layer in S/m
        h1 = m[2] # thickness of the 1st layer
        # Perform the forward function
        Z = EMf_2Lay_HV_field(lambd, sigma1, sigma2, h1, height, offsets, freq, filt)
        return Z               
    def createStartModel(self, dataVals):
        m0 = self.m0
        return m0
