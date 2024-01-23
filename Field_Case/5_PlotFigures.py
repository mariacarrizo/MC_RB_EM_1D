# Script to generate Figure 17 and Table 6:
# Lines from field case

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import copy
import matplotlib
import sys
sys.path.insert(1, '../src')

# Import forward function for 2-layered 1D models in field case
from EM1D import EMf_2Lay_HV_field

# Load estimated models
model_GS = np.load('results/model_2Lay_GS_field.npy')
model_opt = np.load('results/model_2Lay_Opt_field.npy')
model_GSplusOpt = np.load('results/model_2Lay_GSplusOpt_field.npy')

# Load field data
Dataframe = pd.DataFrame(np.load('data/Field_data.npy'),
                        columns = ['X','Y','Position','Z','H2Q','H4Q','H8Q',
                                   'V2Q','V4Q','V8Q','P2Q','P4Q','P8Q',
                                   'H4IP','H8IP','V4IP','V8IP'])

# Position elevations
elev = np.array(Dataframe['Z'])

# Data used for the estimations
data_true = np.array(pd.concat([Dataframe.loc[:,'H2Q':'V8Q'], 
                                      Dataframe.loc[:,'H4IP':]], axis=1))

# Load survey parameters
survey = np.load('data/survey_field.npy', allow_pickle=True).item()
offsets = survey['offsets']
height = survey['height']
freq = survey['freq']
lambd = survey['lambd']
filt = survey['filt']

def grid_elev(model, elev, depthmax=10, distance=37):
    """ Generates a grid from the model to plot a 2D section
    including elevation 
    
    Input:
        model: 1D models to plot
        elev: topographic elevation for each position
        depthmax: maximum depth of the 2D section in m
        distance: maximum distance of the 2D section
        
    Output:
        grid: a grid of size (npos, ny) with the electrical conductivity
        values of the 1D models
    """
    # Arrays for plotting
    depthmax_ = -np.min(elev) + depthmax 
    npos = np.shape(model)[0] # number of 1D models
    ny = 81 # size of the grid in y direction
    y = np.linspace(0, depthmax_, ny) # y axis [m]
    grid = np.zeros((npos, ny)) # empty grid
    sig = model[:,:2].copy() # define electrical conductivities
    thk = model[:,2].copy() - elev # define thicknesses
    
    # Fill the grid with the conductivity values
    for i in range(npos):
        y1 = 0
        y2 = 0
        # Air layer
        while y[y1] <= -elev[i]:
            grid[i, y1] = 0
            y1 += 1
        # First layer
        while y[y1] <= thk[i]:
            grid[i, y1] = sig[i, 0]
            y1 += 1
            y2 = y1
        # Second layer
        grid[i, y2:] = sig[i, 1]
        
    return grid

def rmse_a(p, o):
    """ Calculates the root mean squared error of an array
    p -> predicted
    o -> observed """
    error = np.sqrt(np.sum((p-o)**2))/len(p)
    return error

# Review lines

# Define positions
Line1_pos = np.linspace(0,49,49,endpoint=False,dtype=int)
Line2_pos = np.linspace(900,949,949,endpoint=False,dtype=int)
Line3_pos = np.linspace(1650,1699,1699,endpoint=False, dtype=int)

# Line 1 grids
Line1_GS_grd = grid_elev(model_GS[Line1_pos,:], elev[Line1_pos])
Line1_opt_grd = grid_elev(model_opt[Line1_pos,:], elev[Line1_pos])
Line1_GSplusOpt_grd = grid_elev(model_GSplusOpt[Line1_pos,:], elev[Line1_pos])

# Line 2 grids
Line2_GS_grd = grid_elev(model_GS[Line2_pos,:], elev[Line2_pos])
Line2_opt_grd = grid_elev(model_opt[Line2_pos,:], elev[Line2_pos])
Line2_GSplusOpt_grd = grid_elev(model_GSplusOpt[Line2_pos,:], elev[Line2_pos])

# Line 3 grids
Line3_GS_grd = grid_elev(model_GS[Line3_pos,:], elev[Line3_pos])
Line3_opt_grd = grid_elev(model_opt[Line3_pos,:], elev[Line3_pos])
Line3_GSplusOpt_grd = grid_elev(model_GSplusOpt[Line3_pos,:], elev[Line3_pos])

# Plot
fig, ax = plt.subplots(3,3, sharex=True, sharey=True, layout='constrained',
                      figsize=(6,3))

vmin=10 # mS/m
vmax=180 # mS/m
cmap='viridis'
interp='none'
npos = 51 # Each line has 51 positions
ext=[0,37,10,0] # Each line has 37 m X distance and 10 m depth
norm= colors.LogNorm(vmin=vmin, vmax=vmax)
fs=7 # fontsize

my_cmap = copy.copy(matplotlib.cm.get_cmap('viridis')) # copy the default cmap
my_cmap.set_bad('w')

# Line 1
mod = ax[0,0].imshow((Line1_GS_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,0].set_title('Line 1 GS', fontsize=fs)
ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
ax[0,0].tick_params(labelsize=fs)
ax[0,0].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[0,0].text(34,-1.5,'NE', color='k',fontsize=fs)

ax[0,1].imshow((Line1_opt_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,1].set_title('Line 1 Opt', fontsize=fs)
ax[0,1].tick_params(labelsize=fs)
ax[0,1].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[0,1].text(34,-1.5,'NE', color='k',fontsize=fs)

ax[0,2].imshow((Line1_GSplusOpt_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,2].set_title('Line 1 GS + Opt', fontsize=fs)
ax[0,2].tick_params(labelsize=fs)
ax[0,2].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[0,2].text(34,-1.5,'NE', color='k',fontsize=fs)

#Line 2
ax[1,0].imshow((Line2_GS_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[1,0].set_title('Line 2 GS', fontsize=fs)
ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
ax[1,0].tick_params(labelsize=fs)
ax[1,0].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[1,0].text(34,-1.5,'NE', color='k',fontsize=fs)

ax[1,1].imshow((Line2_opt_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[1,1].set_title('Line 2 Opt', fontsize=fs)
ax[1,1].tick_params(labelsize=fs)
ax[1,1].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[1,1].text(34,-1.5,'NE', color='k',fontsize=fs)

ax[1,2].imshow((Line2_GSplusOpt_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[1,2].set_title('Line 2 GS + Opt', fontsize=fs)
ax[1,2].tick_params(labelsize=fs)
ax[1,2].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[1,2].text(34,-1.5,'NE', color='k',fontsize=fs)

# Line 3
ax[2,0].imshow((Line3_GS_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[2,0].set_title('Line 3 GS', fontsize=fs)
ax[2,0].set_ylabel('Depth [m]', fontsize=fs)
ax[2,0].tick_params(labelsize=fs)
ax[2,0].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[2,0].text(34,-1.5,'NE', color='k',fontsize=fs)
ax[2,0].set_xlabel('Distance [m]', fontsize=fs)

ax[2,1].imshow((Line3_opt_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[2,1].set_title('Line 3 Opt', fontsize=fs)
ax[2,1].tick_params(labelsize=fs)
ax[2,1].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[2,1].text(34,-1.5,'NE', color='k',fontsize=fs)
ax[2,1].set_xlabel('Distance [m]', fontsize=fs)

ax[2,2].imshow((Line3_GSplusOpt_grd*1000).T, cmap=my_cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[2,2].set_title('Line 3 GS + Opt', fontsize=fs)
ax[2,2].tick_params(labelsize=fs)
ax[2,2].text(-0.5,-1.5,'SW', color='k',fontsize=fs)
ax[2,2].text(34,-1.5,'NE', color='k',fontsize=fs)
ax[2,2].set_xlabel('Distance [m]', fontsize=fs)

clb_mod = fig.colorbar(mod, ax=ax[:,2], location='right', shrink=0.6)
clb_mod.ax.tick_params(labelsize=fs)
clb_mod.set_label('$\sigma$ [mS/m]', fontsize=fs )

ax[0,0].set_aspect(aspect=1.5)
ax[0,1].set_aspect(aspect=1.5)
ax[0,2].set_aspect(aspect=1.5)

ax[1,0].set_aspect(aspect=1.5)
ax[1,1].set_aspect(aspect=1.5)
ax[1,2].set_aspect(aspect=1.5)

ax[2,0].set_aspect(aspect=1.5)
ax[2,1].set_aspect(aspect=1.5)
ax[2,2].set_aspect(aspect=1.5)
plt.savefig('figures/Field_Lines_2Lay.eps', format='eps')

#%%
# Simulate data for each model

data_GS = []
data_opt = []
data_GSplusOpt = []

for pos in range(len(data_true)):
    
    data_GS.append(EMf_2Lay_HV_field(lambd,
                                      sigma1 = model_GS[pos,0],
                                      sigma2 = model_GS[pos,1],
                                      h1 = model_GS[pos,2],
                                      height = height,
                                      offsets = offsets,
                                      freq = freq,
                                      filt = filt))

    data_opt.append(EMf_2Lay_HV_field(lambd,
                                      sigma1 = model_opt[pos,0],
                                      sigma2 = model_opt[pos,1],
                                      h1 = model_opt[pos,2],
                                      height = height,
                                      offsets = offsets,
                                      freq = freq,
                                      filt = filt))

    data_GSplusOpt.append(EMf_2Lay_HV_field(lambd,
                                      sigma1 = model_GSplusOpt[pos,0],
                                      sigma2 = model_GSplusOpt[pos,1],
                                      h1 = model_GSplusOpt[pos,2],
                                      height = height,
                                      offsets = offsets,
                                      freq = freq,
                                      filt = filt))
    
# Transform data to parts per thousand [ppt]

data_GS = np.array(data_GS)*1e3
data_opt = np.array(data_opt)*1e3
data_GSplusOpt = np.array(data_GSplusOpt)*1e3
data_true = data_true*1e3

# Calculate RMSE for each method

RMSE_GS= rmse_a(data_GS, data_true)
RMSE_opt = rmse_a(data_opt, data_true)
RMSE_GSplusOpt = rmse_a(data_GSplusOpt, data_true)

# Print RMSE in ppt

RMSE = pd.DataFrame([RMSE_GS, RMSE_opt, RMSE_GSplusOpt],
                    index= ['GS', 'Opt', 'GSplusOpt'],
                   columns=['RMSE [ppt]'])

print('Table 6')
print(RMSE)