# Script to perform global search inversion for Field case

# Import libraries
import numpy as np
import time
import pandas as pd
from scipy.constants import mu_0
import sys
sys.path.insert(1, '../src')

# Load function that performs global search in lookup table
from EM1D import GlobalSearch_2Lay

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

# Load lookup table and sampling 
LUT = np.load('data/LUTable_2Lay_Texel.npy')
conds = np.load('data/conds.npy')
thicks =  np.load('data/thicks.npy')

# Load field data 
Dataframe = pd.read_csv('data/DualemTexel_27-03-2024.csv', 
                        names = ['Time','Altitude','Latitude','Longitude',
                                   'HDOP','Counter','Voltage','Current',
                                   'Pdeg','Rdeg','DualemTimei2','H2mS_m',
                                   'H2ppt','P2mS_m','P2ppt','DualemTimei4',
                                   'H4mS_m','H4ppt','P4mS_m','P4ppt',
                                   'DualemTimei8','H8mS_m','H8ppt','P8mS_m','P8ppt'])

# Obtain Quadrature and In-Phase values

H2Q = Q_from_Sigma(Dataframe['H2mS_m']/1000, 2)
P2Q = Q_from_Sigma(Dataframe['P2mS_m']/1000, 2.1)
H4Q = Q_from_Sigma(Dataframe['H4mS_m']/1000, 4)
P4Q = Q_from_Sigma(Dataframe['P4mS_m']/1000, 4.1)
H8Q = Q_from_Sigma(Dataframe['H8mS_m']/1000, 8)
P8Q = Q_from_Sigma(Dataframe['P8mS_m']/1000, 8.1)

H2IP = Dataframe['H2ppt']/1000
P2IP = Dataframe['P2ppt']/1000
H4IP = Dataframe['H4ppt']/1000
P4IP = Dataframe['P4ppt']/1000
H8IP = Dataframe['H8ppt']/1000
P8IP = Dataframe['P8ppt']/1000

# Obtain H and P quadrature and in-phase measurements

data = np.array(pd.concat([H2Q, H4Q, H8Q, H2IP, H4IP, H8IP, P2Q, P4Q, P8Q, P2IP, P4IP, P8IP], axis=1))

# number of 1D models
npos = len(data)

# Estimate with both Quadrature and In Phase
model_est = [] # Empty array for estimated model

print('Starting global search ...')

starttime = time.time()
for p in range(npos):
    model_est.append(GlobalSearch_2Lay(LUT, data[p], conds, thicks, nsl=len(conds)))
endtime = time.time() - starttime

print('Global search Q+IP excution for ', npos, ' positions: ', f"{(endtime/60):.3}", 'minutes')

# Save estimated models
np.save('results/model_2Lay_GS_Texel', model_est)
