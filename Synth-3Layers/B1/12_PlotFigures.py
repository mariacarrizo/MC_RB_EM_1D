# Script to generate Figures and Tables

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import copy
import matplotlib

# Functions

def grid(model, depthmax=10):
    """ Generates a grid from the model to plot a 2D section"""
    # Arrays for plotting
    npos = np.shape(model)[0] # number of 1D models
    ny = 71 # size of the grid in y direction
    y = np.linspace(0, depthmax, ny) # y axis [m]
    grid = np.zeros((npos, ny)) # empty grid
    sig = model[:,:3].copy() # define electrical conductivities
    thk = model[:,3:].copy()  # define thicknesses
    
    # Fill the grid with the conductivity values
    for i in range(npos):
        y1 = 0
        # First layer
        while y[y1] <= thk[i,0]:
            grid[i, y1] = sig[i, 0]
            y1 += 1
            y2 = y1
        # Second layer
        while y[y2] <= (thk[i,0] + thk[i,1]):
            grid[i, y2] = sig[i, 1]
            y2 += 1
        # Third layer
        grid[i, y2:] = sig[i, 2]
        
    return grid

def rmse_a(p, o):
    """ Calculates the root mean squared error of an array
    p -> predicted
    o -> observed """
    error = np.sqrt(np.sum((p-o)**2))/len(p)
    return error

# Load models
model = np.load('data/model_synth_3Lay_B1.npy') # True model

# Models estimated with global search
model_GS = np.load('results/model_3Lay_B1_GS.npy')
model_GS_Q = np.load('results/model_3Lay_B1_GS_Q.npy')
model_GS_IP = np.load('results/model_3Lay_B1_GS_IP.npy')

# Models estimated with gradient descent
model_opt = np.load('results/model_3Lay_B1_Opt.npy')
model_opt_Q = np.load('results/model_3Lay_B1_Opt_Q.npy')
model_opt_IP = np.load('results/model_3Lay_B1_Opt_IP.npy')
                        
# Model estimated using global search + optimization
model_GSplusOpt = np.load('results/model_3Lay_GSplusOpt_B1.npy')

# Create model grids (for plotting)
model_grd = grid(model)

model_GS_grd = grid(model_GS)
model_GS_Q_grd = grid(model_GS_Q)
model_GS_IP_grd = grid(model_GS_IP)

model_opt_grd = grid(model_opt)
model_opt_Q_grd = grid(model_opt_Q)
model_opt_IP_grd = grid(model_opt_IP)

model_GSplusOpt_grd = grid(model_GSplusOpt)

# Create error grids (for plotting)
error_GS = np.abs((model_grd-model_GS_grd)/model_grd)*100
error_GS_Q = np.abs((model_grd-model_GS_Q_grd)/model_grd)*100
error_GS_IP = np.abs((model_grd-model_GS_IP_grd)/model_grd)*100

error_opt = np.abs((model_grd-model_opt_grd)/model_grd)*100
error_opt_Q = np.abs((model_grd-model_opt_Q_grd)/model_grd)*100
error_opt_IP = np.abs((model_grd-model_opt_IP_grd)/model_grd)*100

error_GSplusOpt = np.abs((model_grd - model_GSplusOpt_grd)/model_grd)*100

#%%
# Figure 8

fig, ax = plt.subplots(5,3, sharex=True, sharey=True, layout='constrained')

depth_true1 = model[:,3].copy()
depth_true2 = model[:,3] + model[:,4]
vmin=10
vmax=200
cmap='viridis'
interp='none'
npos = len(model)
ext=[0,npos,10,0]
norm= colors.LogNorm(vmin=vmin, vmax=vmax)
fs=7 # fontsize

my_cmap = copy.copy(matplotlib.cm.get_cmap("RdBu_r", 8)) # copy the default cmap
my_cmap.set_bad('#01386a')

x = np.linspace(0,npos+1,npos+1, endpoint=False)
ax[1,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')
ax[1,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')
ax[1,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')

ax[3,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')
ax[3,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')
ax[3,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')

ax[1,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')
ax[1,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')
ax[1,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')

ax[3,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')
ax[3,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')
ax[3,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')

ax[0,0].remove()
ax[0,2].remove()
mod = ax[0,1].imshow((model_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,1].set_title('True Model', fontsize=fs)
ax[0,1].set_ylabel('Depth [m]', fontsize=fs)
ax[0,1].tick_params(labelsize=fs)
ax[0,1].text(-0.5,-1.5,'a)', color='k',fontsize=fs)

ax[1,0].imshow((model_GS_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[1,0].set_title('Global search [Q+IP]', fontsize=fs)
ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
ax[1,0].tick_params(labelsize=fs)
ax[1,0].text(-0.5,-1.5,'b)', color='k',fontsize=fs)

ax[1,1].imshow((model_GS_Q_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[1,1].set_title('Global search [Q]', fontsize=fs)
ax[1,1].text(-0.5,-1.5,'c)', color='k',fontsize=fs)

ax[1,2].imshow((model_GS_IP_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[1,2].set_title('Global search [IP]', fontsize=fs)
ax[1,2].text(-0.5,-1.5,'d)', color='k',fontsize=fs)

err = ax[2,0].imshow(np.log10(error_GS).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[2,0].set_title('Relative error [Q+IP]', fontsize=fs)
ax[2,0].set_ylabel('Depth [m]', fontsize=fs)
ax[2,0].tick_params(labelsize=fs)
ax[2,0].text(-0.5,-1.5,'e)', color='k',fontsize=fs)

ax[2,1].imshow(np.log10(error_GS_Q).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[2,1].set_title('Relative error [Q]', fontsize=fs)
ax[2,1].text(-0.5,-1.5,'f)', color='k',fontsize=fs)

ax[2,2].imshow(np.log10(error_GS_IP).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[2,2].set_title('Relative error [IP]', fontsize=fs)
ax[2,2].text(-0.5,-1.5,'g)', color='k',fontsize=fs)

ax[3,0].imshow((model_opt_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[3,0].set_title('Optimization [Q+IP]', fontsize=fs)
ax[3,0].set_ylabel('Depth [m]', fontsize=fs)
ax[3,0].tick_params(labelsize=fs)
ax[3,0].text(-0.5,-1.5,'h)', color='k',fontsize=fs)

ax[3,1].imshow((model_opt_Q_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[3,1].set_title('Optimization [Q]', fontsize=fs)
ax[3,1].text(-0.5,-1.5,'i)', color='k',fontsize=fs)

ax[3,2].imshow((model_opt_IP_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[3,2].set_title('Optimization [IP]', fontsize=fs)
ax[3,2].text(-0.5,-1.5,'j)', color='k',fontsize=fs)

ax[4,0].imshow(np.log10(error_opt).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[4,0].set_title('Relative error [Q+IP]', fontsize=fs)
ax[4,0].set_ylabel('Depth [m]', fontsize=fs)
ax[4,0].set_xlabel('Distance [m]', fontsize=fs)
ax[4,0].tick_params(labelsize=fs)
ax[4,0].text(-0.5,-1.5,'k)', color='k',fontsize=fs)

ax[4,1].imshow(np.log10(error_opt_Q).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[4,1].set_title('Relative error [Q]', fontsize=fs)
ax[4,1].set_xlabel('Distance [m]', fontsize=fs)
ax[4,1].tick_params(labelsize=fs)
ax[4,1].text(-0.5,-1.5,'l)', color='k',fontsize=fs)

ax[4,2].imshow(np.log10(error_opt_IP).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[4,2].set_title('Relative error [IP]', fontsize=fs)
ax[4,2].set_xlabel('Distance [m]', fontsize=fs)
ax[4,2].tick_params(labelsize=fs)
ax[4,2].text(-0.5,-1.5,'m)', color='k',fontsize=fs)

clb_mod = fig.colorbar(mod, ax=ax[1:,2], location='right', shrink=0.6)
clb_mod.ax.tick_params(labelsize=fs)
clb_mod.set_label('$\sigma$ [mS/m]', fontsize=fs )

clb_err = fig.colorbar(err, ax=ax[1:,0], location='left', shrink=0.6)
clb_err.ax.tick_params(labelsize=fs)
clb_err.set_label('Relative error', fontsize=fs )
clb_err.set_ticks([ 0, 1, 2])
clb_err.ax.set_yticklabels([ r"$1\,\%$",
                         r"$10\,\%$", r"$100\,\%$"])
plt.savefig('figures/3Lay_B1_GSvsOpt.eps', format='eps')

#%%
# Figure 14

fig, ax = plt.subplots(2,3, sharex=True, sharey=True, layout='constrained', 
                       figsize=(6,2.5))

depth_true1 = model[:,3].copy()
depth_true2 = model[:,3] + model[:,4]
vmin=10
vmax=200
cmap='viridis'
interp='none'
npos = len(model)
ext=[0,npos,10,0]
norm= colors.LogNorm(vmin=vmin, vmax=vmax)
fs=7 # fontsize

x = np.linspace(0,npos+1,npos+1, endpoint=False)
ax[0,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')
ax[0,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')
ax[0,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true1[0],depth_true1[:], depth_true1[-2])), ':r')

ax[0,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')
ax[0,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')
ax[0,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true2[0],depth_true2[:], depth_true2[-2])), ':r')

ax[0,0].imshow((model_GS_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,0].set_title('Global search', fontsize=fs)
ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
ax[0,0].tick_params(labelsize=fs)
ax[0,0].text(-0.5,-1.5,'a)', color='k',fontsize=fs)

ax[0,1].imshow((model_opt_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,1].set_title('Optimization', fontsize=fs)
ax[0,1].tick_params(labelsize=fs)
ax[0,1].text(-0.5,-1.5,'b)', color='k',fontsize=fs)

ax[0,2].imshow((model_GSplusOpt_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,2].set_title('GS + Opt', fontsize=fs)
ax[0,2].tick_params(labelsize=fs)
ax[0,2].text(-0.5,-1.5,'c)', color='k',fontsize=fs)

err = ax[1,0].imshow(np.log10(error_GS).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[1,0].set_title('Relative error (GS)', fontsize=fs)
ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
ax[1,0].tick_params(labelsize=fs)
ax[1,0].text(-0.5,-1.5,'d)', color='k',fontsize=fs)
ax[1,0].set_xlabel('Distance [m]', fontsize=fs)

ax[1,1].imshow(np.log10(error_opt).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[1,1].set_title('Relative error (Opt)', fontsize=fs)
ax[1,1].tick_params(labelsize=fs)
ax[1,1].text(-0.5,-1.5,'e)', color='k',fontsize=fs)
ax[1,1].set_xlabel('Distance [m]', fontsize=fs)

ax[1,2].imshow(np.log10(error_GSplusOpt).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=my_cmap)
ax[1,2].set_title('Relative error (GS+Opt)', fontsize=fs)
ax[1,2].tick_params(labelsize=fs)
ax[1,2].text(-0.5,-1.5,'f)', color='k',fontsize=fs)
ax[1,1].set_xlabel('Distance [m]', fontsize=fs)

clb_mod = fig.colorbar(mod, ax=ax[0:,2], location='right', shrink=0.6)
clb_mod.ax.tick_params(labelsize=fs)
clb_mod.set_label('$\sigma$ [mS/m]', fontsize=fs )

clb_err = fig.colorbar(err, ax=ax[0:,0], location='left', shrink=0.6)
clb_err.ax.tick_params(labelsize=fs)
clb_err.set_label('Relative error', fontsize=fs )
clb_err.set_ticks([ 0, 1, 2])
clb_err.ax.set_yticklabels([ r"$1\,\%$",
                         r"$10\,\%$", r"$100\,\%$"])
plt.savefig('figures/3Lay_B1_GSplusOpt.eps', format='eps')

#%%
# Table 5 Case B.1

RMSE_s_GS = rmse_a(np.hstack(model[:,:3]), np.hstack(model_GS[:,:3]))*1000
RMSE_h_GS = rmse_a(np.hstack(model[:,3:]), np.hstack(model_GS[:,3:]))

RMSE_s_GS_Q = rmse_a(np.hstack(model[:,:3]), np.hstack(model_GS_Q[:,:3]))*1000
RMSE_h_GS_Q = rmse_a(np.hstack(model[:,3:]), np.hstack(model_GS_Q[:,3:]))

RMSE_s_GS_IP = rmse_a(np.hstack(model[:,:3]), np.hstack(model_GS_IP[:,:3]))*1000
RMSE_h_GS_IP = rmse_a(np.hstack(model[:,3:]), np.hstack(model_GS_IP[:,3:]))

RMSE_s_Opt = rmse_a(np.hstack(model[:,:3]), np.hstack(model_opt[:,:3]))*1000
RMSE_h_Opt = rmse_a(np.hstack(model[:,3:]), np.hstack(model_opt[:,3:]))

RMSE_s_Opt_Q = rmse_a(np.hstack(model[:,:3]), np.hstack(model_opt_Q[:,:3]))*1000
RMSE_h_Opt_Q = rmse_a(np.hstack(model[:,3:]), np.hstack(model_opt_Q[:,3:]))

RMSE_s_Opt_IP = rmse_a(np.hstack(model[:,:3]), np.hstack(model_opt_IP[:,:3]))*1000
RMSE_h_Opt_IP = rmse_a(np.hstack(model[:,3:]), np.hstack(model_opt_IP[:,3:]))

RMSE_s_GSplusOpt = rmse_a(np.hstack(model[:,:3]), np.hstack(model_GSplusOpt[:,:3]))*1000
RMSE_h_GSplusOpt = rmse_a(np.hstack(model[:,3:]), np.hstack(model_GSplusOpt[:,3:]))

Error_List_Method = [['Global search [Q+IP]', RMSE_s_GS, RMSE_h_GS],
              ['Global search [Q]', RMSE_s_GS_Q, RMSE_h_GS_Q],
              ['Global search [IP]', RMSE_s_GS_IP, RMSE_h_GS_IP],
              ['Optimization [Q+IP]', RMSE_s_Opt, RMSE_h_Opt],
              ['Optimization [Q]', RMSE_s_Opt_Q, RMSE_h_Opt_Q],
              ['Optimization [IP]', RMSE_s_Opt_IP, RMSE_h_Opt_IP],
              ['GS + Opt [Q + IP]', RMSE_s_GSplusOpt, RMSE_h_GSplusOpt]]

RMSE = pd.DataFrame(Error_List_Method, columns=['Method', 'RMSE $\sigma$ [mS/m]', 'RMSE $h$ [m]'])

print('Table 5 Case B.1')
print(RMSE)
print()


