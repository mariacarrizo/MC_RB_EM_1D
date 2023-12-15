import numpy as np
from scipy.constants import mu_0

# Functions to calculate mutual impedance ratios

def Z_H(s, R_0, lambd, a, filt):
    Z = - s**3 * (np.dot((R_0*np.exp(-2*lambd*a)*lambd**2), filt.j0)/s)
    return Z

def Z_V(s, R_0, lambd, a, filt):
    Z = - s**2 * (np.dot((R_0*np.exp(-2*lambd*a)*lambd), filt.j1)/s)
    return Z

def Z_P(s, R_0, lambd, a, filt):
    Z = s**3 * (np.dot((R_0*np.exp(-2*lambd*a)*lambd**2),filt.j1)/s)
    return Z

def R0_2Lay(lam, sigma1, sigma2, h1, freq):
    """ Recursive function to calculate reflection coefficient R_0 for a 2-layered model 
    """
    gam1 = np.sqrt(lam**2 + 1j*(2 * np.pi * freq) *sigma1 * mu_0)
    gam2 = np.sqrt(lam**2 + 1j*(2 * np.pi * freq) *sigma2 * mu_0)
    r1 = (gam1 - gam2)/(gam1 + gam2)
    r0 = ((lam - gam1)/(lam + gam1) + r1 * np.exp(- 2 * gam1 * h1))/(1 + (lam - gam1)/(lam
           + gam1) * r1 * np.exp(-2 * gam1 * h1))
    return r0

def R0_3Lay(lam, sigma1, sigma2, sigma3, h1, h2, freq):
    """ Recursive function to calculate reflection coefficient R_0 for a 3-layered model
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
    """
    # Calculate reflection coefficient
    R0 = R0_2Lay(lambd, sigma1, sigma2, h1, freq)
    # Calculate mutual impedance ratios for each coil-coil geometry
    Z_h = Z_H(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_v = Z_V(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    Z_p = Z_P(s=offsets, R_0= R0, lambd=lambd, a=height, filt=filt)
    # Obtain quadratures
    Q_h = Z_h.imag
    Q_v = Z_v.imag
    Q_p = Z_p.imag
    # Obtain in-phases
    IP_h = Z_h.real
    IP_v = Z_v.real
    IP_p = Z_p.real
    
    return np.hstack((Q_h, Q_v, Q_p, IP_h, IP_v, IP_p))
    
def GlobalSearch(Database, Data, conds, thicks, nsl=51):
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

