# Generate Figures and tables

# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd

# Functions

def grid(model, depthmax=10):
    """ Generates a grid from the model to plot a 2D section"""
    # Arrays for plotting   
    npos = np.shape(model)[0] # number of 1D models
    ny = 81 # size of the grid in y direction
    y = np.linspace(0, depthmax, ny) # y axis [m]
    grid = np.zeros((npos, ny)) # empty grid
    sig = model[:,:2].copy() # define electrical conductivities
    thk = model[:,2].copy()  # define thicknesses
    
    # Fill the grid with the conductivity values
    for i in range(npos):
        y1 = 0
        # First layer
        while y[y1] <= thk[i]:
            grid[i, y1] = sig[i, 0]
            y1 += 1
        # Second layer
        grid[i, y1:] = sig[i, 1]
        
    return grid

def rmse_a(p, o):
    """ Calculates the root mean squared error of an array
    p -> predicted
    o -> observed """
    error = np.sqrt(np.sum((p-o)**2))/len(p)
    return error

# Load models
model = np.load('data/model_synth_2Lay_A2.npy')

# Models estimated with global search
model_GS = np.load('results/model_2Lay_A2_GS.npy')
model_GS_Q = np.load('results/model_2Lay_A2_GS_Q.npy')
model_GS_IP = np.load('results/model_2Lay_A2_GS_IP.npy')

# Models estimated with gradient descent
model_opt = np.load('results/model_2Lay_A2_Opt.npy')
model_opt_Q = np.load('results/model_2Lay_A2_Opt_Q.npy')
model_opt_IP = np.load('results/model_2Lay_A2_Opt_IP.npy')
                        
# Models estimated with global search, data with added noise
model_GS_n2 = np.load('results/model_2Lay_A2_GS_n2.npy')
model_GS_n5 = np.load('results/model_2Lay_A2_GS_n5.npy')
model_GS_n10 = np.load('results/model_2Lay_A2_GS_n10.npy')

# Models estimated with gradient descent, data with added noise
model_opt_n2 = np.load('results/model_2Lay_A2_Opt_n2.npy')
model_opt_n5 = np.load('results/model_2Lay_A2_Opt_n5.npy')
model_opt_n10 = np.load('results/model_2Lay_A2_Opt_n10.npy')

# Create model grids (for plotting)
model_grd = grid(model)

model_GS_grd = grid(model_GS)
model_GS_Q_grd = grid(model_GS_Q)
model_GS_IP_grd = grid(model_GS_IP)

model_opt_grd = grid(model_opt)
model_opt_Q_grd = grid(model_opt_Q)
model_opt_IP_grd = grid(model_opt_IP)

model_GS_n2_grd = grid(model_GS_n2)
model_GS_n5_grd = grid(model_GS_n5)
model_GS_n10_grd = grid(model_GS_n10)

model_opt_n2_grd = grid(model_opt_n2)
model_opt_n5_grd = grid(model_opt_n5)
model_opt_n10_grd = grid(model_opt_n10)

# Create error grids (for plotting)
error_GS = np.abs((model_grd-model_GS_grd)/model_grd)*100
error_GS_Q = np.abs((model_grd-model_GS_Q_grd)/model_grd)*100
error_GS_IP = np.abs((model_grd-model_GS_IP_grd)/model_grd)*100

error_GS_n2 = np.abs((model_grd-model_GS_n2_grd)/model_grd)*100
error_GS_n5 = np.abs((model_grd-model_GS_n5_grd)/model_grd)*100
error_GS_n10 = np.abs((model_grd-model_GS_n10_grd)/model_grd)*100

error_opt = np.abs((model_grd-model_opt_grd)/model_grd)*100
error_opt_Q = np.abs((model_grd-model_opt_Q_grd)/model_grd)*100
error_opt_IP = np.abs((model_grd-model_opt_IP_grd)/model_grd)*100

error_opt_n2 = np.abs((model_grd-model_opt_n2_grd)/model_grd)*100
error_opt_n5 = np.abs((model_grd-model_opt_n5_grd)/model_grd)*100
error_opt_n10 = np.abs((model_grd-model_opt_n10_grd)/model_grd)*100

#%%
# Figure 3

fig, ax = plt.subplots(5,3, sharex=True, sharey=True, layout='constrained')

depth_true = model[:,2].copy()
vmin=10
vmax=300
cmap='viridis'
interp='none'
npos = len(model)
ext=[0,npos,10,0]
norm= colors.LogNorm(vmin=vmin, vmax=vmax)
fs=7

x = np.linspace(0,npos+1,npos+1, endpoint=False)
ax[1,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[1,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[1,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')

ax[3,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[3,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[3,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')

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
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[2,0].set_title('Relative error [Q+IP]', fontsize=fs)
ax[2,0].set_ylabel('Depth [m]', fontsize=fs)
ax[2,0].tick_params(labelsize=fs)
ax[2,0].text(-0.5,-1.5,'e)', color='k',fontsize=fs)

ax[2,1].imshow(np.log10(error_GS_Q).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[2,1].set_title('Relative error [Q]', fontsize=fs)
ax[2,1].text(-0.5,-1.5,'f)', color='k',fontsize=fs)

ax[2,2].imshow(np.log10(error_GS_IP).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
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
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[4,0].set_title('Relative error [Q+IP]', fontsize=fs)
ax[4,0].set_ylabel('Depth [m]', fontsize=fs)
ax[4,0].set_xlabel('Distance [m]', fontsize=fs)
ax[4,0].tick_params(labelsize=fs)
ax[4,0].text(-0.5,-1.5,'k)', color='k',fontsize=fs)

ax[4,1].imshow(np.log10(error_opt_Q).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[4,1].set_title('Relative error [Q]', fontsize=fs)
ax[4,1].set_xlabel('Distance [m]', fontsize=fs)
ax[4,1].tick_params(labelsize=fs)
ax[4,1].text(-0.5,-1.5,'l)', color='k',fontsize=fs)

ax[4,2].imshow(np.log10(error_opt_IP).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
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
plt.savefig('figures/2Lay_UppCond_GSvsOpt.eps', format='eps')

#%%
# Figure 7

fig, ax = plt.subplots(4,3, sharex=True, sharey=True, layout='constrained')

ax[0,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[0,0].imshow((model_GS_n2_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,0].set_title('Global search, $\epsilon$: 2.5%', fontsize=fs)
ax[0,0].set_ylabel('Depth [m]', fontsize=fs)
ax[0,0].tick_params(labelsize=fs)
ax[0,0].text(-0.5,-1.5,'a)', color='k',fontsize=fs)

ax[0,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[0,1].imshow((model_GS_n5_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,1].set_title('Global search, $\epsilon$: 5%', fontsize=fs)
ax[0,1].text(-0.5,-1.5,'b)', color='k',fontsize=fs)

ax[0,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[0,2].imshow((model_GS_n10_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[0,2].set_title('Global search, $\epsilon$: 10%', fontsize=fs)
ax[0,2].text(-0.5,-1.5,'c)', color='k',fontsize=fs)

ax[1,0].imshow(np.log10(error_GS_n2).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[1,0].set_title('Relative error, $\epsilon$: 2.5%', fontsize=fs)
ax[1,0].set_ylabel('Depth [m]', fontsize=fs)
ax[1,0].tick_params(labelsize=fs)
ax[1,0].text(-0.5,-1.5,'d)', color='k',fontsize=fs)

ax[1,1].imshow(np.log10(error_GS_n5).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[1,1].set_title('Relative error, $\epsilon$: 5%', fontsize=fs)
ax[1,1].text(-0.5,-1.5,'e)', color='k',fontsize=fs)

ax[1,2].imshow(np.log10(error_GS_n10).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[1,2].set_title('Relative error, $\epsilon$: 10%', fontsize=fs)
ax[1,2].text(-0.5,-1.5,'f)', color='k',fontsize=fs)

ax[2,0].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[2,0].imshow((model_opt_n2_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[2,0].set_title('Optimization, $\epsilon$: 2.5%', fontsize=fs)
ax[2,0].set_ylabel('Depth [m]', fontsize=fs)
ax[2,0].tick_params(labelsize=fs)
ax[2,0].text(-0.5,-1.5,'g)', color='k',fontsize=fs)

ax[2,1].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
ax[2,1].imshow((model_opt_n5_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[2,1].set_title('Optimization, $\epsilon$: 5%', fontsize=fs)
ax[2,1].text(-0.5,-1.5,'h)', color='k',fontsize=fs)

ax[2,2].step(np.hstack((x, x[-1])), 
         np.hstack((depth_true[0],depth_true[:], depth_true[-2])), ':r')
mod = ax[2,2].imshow((model_opt_n10_grd*1000).T, cmap=cmap, interpolation=interp, 
                    extent=ext, norm = norm)
ax[2,2].set_title('Optimization, $\epsilon$: 10%', fontsize=fs)
ax[2,2].text(-0.5,-1.5,'i)', color='k',fontsize=fs)

ax[3,0].imshow(np.log10(error_opt_n2).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[3,0].set_title('Relative error, $\epsilon$: 2.5%', fontsize=fs)
ax[3,0].set_ylabel('Depth [m]', fontsize=fs)
ax[3,0].set_xlabel('Distance [m]', fontsize=fs)
ax[3,0].tick_params(labelsize=fs)
ax[3,0].text(-0.5,-1.5,'j)', color='k',fontsize=fs)

ax[3,1].imshow(np.log10(error_opt_n5).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[3,1].set_title('Relative error, $\epsilon$: 5%', fontsize=fs)
ax[3,1].set_xlabel('Distance [m]', fontsize=fs)
ax[3,1].tick_params(labelsize=fs)
ax[3,1].text(-0.5,-1.5,'k)', color='k',fontsize=fs)

err = ax[3,2].imshow(np.log10(error_opt_n10).T, vmin=0, vmax=2, extent=ext, interpolation=interp,
          cmap=plt.cm.get_cmap("RdBu_r", 8))
ax[3,2].set_title('Relative error, $\epsilon$: 10%', fontsize=fs)
ax[3,2].set_xlabel('Distance [m]', fontsize=fs)
ax[3,2].tick_params(labelsize=fs)
ax[3,2].text(-0.5,-1.5,'l)', color='k',fontsize=fs)

clb_mod = fig.colorbar(mod, ax=ax[:,2], location='right', shrink=0.6)
clb_mod.ax.tick_params(labelsize=fs)
clb_mod.set_label('$\sigma$ [mS/m]', fontsize=fs )

clb_err = fig.colorbar(err, ax=ax[:,0], location='left', shrink=0.6)
clb_err.ax.tick_params(labelsize=fs)
clb_err.set_label('Relative error', fontsize=fs )
clb_err.set_ticks([ 0, 1, 2])
clb_err.ax.set_yticklabels([ r"$1\,\%$",
                         r"$10\,\%$", r"$100\,\%$"])

plt.savefig('figures/2Lay_UppCond_GSvsOpt_noise.eps', format='eps')

#%%
# Table 3 case A.2

RMSE_s_GS = rmse_a(np.log10(np.hstack(model[:,:2])), 
                   np.log10(np.hstack(model_GS[:,:2])))*1000
RMSE_h_GS = rmse_a(model[:,2], model_GS[:,2])

RMSE_s_GS_Q = rmse_a(np.log10(np.hstack(model[:,:2])),
                     np.log10(np.hstack(model_GS_Q[:,:2])))*1000
RMSE_h_GS_Q = rmse_a(model[:,2], model_GS_Q[:,2])

RMSE_s_GS_IP = rmse_a(np.log10(np.hstack(model[:,:2])), 
                      np.log10(np.hstack(model_GS_IP[:,:2])))*1000
RMSE_h_GS_IP = rmse_a(model[:,2], model_GS_IP[:,2])

RMSE_s_Opt = rmse_a(np.log10(np.hstack(model[:,:2])),
                    np.log10(np.hstack(model_opt[:,:2])))*1000
RMSE_h_Opt = rmse_a(model[:,2], model_opt[:,2])

RMSE_s_Opt_Q = rmse_a(np.log10(np.hstack(model[:,:2])),
                      np.log10(np.hstack(model_opt_Q[:,:2])))*1000
RMSE_h_Opt_Q = rmse_a(model[:,2], model_opt_Q[:,2])

RMSE_s_Opt_IP = rmse_a(np.log10(np.hstack(model[:,:2])),
                       np.log10(np.hstack(model_opt_IP[:,:2])))*1000
RMSE_h_Opt_IP = rmse_a(model[:,2], model_opt_IP[:,2])

Error_List_Method = [['Global search [Q+IP]', RMSE_s_GS, RMSE_h_GS],
              ['Global search [Q]', RMSE_s_GS_Q, RMSE_h_GS_Q],
              ['Global search [IP]', RMSE_s_GS_IP, RMSE_h_GS_IP],
              ['Optimization [Q+IP]', RMSE_s_Opt, RMSE_h_Opt],
              ['Optimization [Q]', RMSE_s_Opt_Q, RMSE_h_Opt_Q],
              ['Optimization [IP]', RMSE_s_Opt_IP, RMSE_h_Opt_IP]]

RMSE = pd.DataFrame(Error_List_Method, columns=['Method', 'RMSE $\sigma$ [mS/m]', 'RMSE $h$ [m]'])

print('Table 3 Case A.2')
print(RMSE)
print()

#%%
# Table 4 case A.2

RMSE_s_GS_n2 = rmse_a(np.log10(np.hstack(model[:,:2])), 
                   np.log10(np.hstack(model_GS_n2[:,:2])))*1000
RMSE_h_GS_n2 = rmse_a(model[:,2], model_GS_n2[:,2])

RMSE_s_GS_n5 = rmse_a(np.log10(np.hstack(model[:,:2])), 
                   np.log10(np.hstack(model_GS_n5[:,:2])))*1000
RMSE_h_GS_n5 = rmse_a(model[:,2], model_GS_n5[:,2])

RMSE_s_GS_n10 = rmse_a(np.log10(np.hstack(model[:,:2])), 
                   np.log10(np.hstack(model_GS_n10[:,:2])))*1000
RMSE_h_GS_n10 = rmse_a(model[:,2], model_GS_n10[:,2])

RMSE_s_Opt_n2 = rmse_a(np.log10(np.hstack(model[:,:2])),
                    np.log10(np.hstack(model_opt_n2[:,:2])))*1000
RMSE_h_Opt_n2 = rmse_a(model[:,2], model_opt_n2[:,2])

RMSE_s_Opt_n5 = rmse_a(np.log10(np.hstack(model[:,:2])),
                    np.log10(np.hstack(model_opt_n5[:,:2])))*1000
RMSE_h_Opt_n5 = rmse_a(model[:,2], model_opt_n5[:,2])

RMSE_s_Opt_n10 = rmse_a(np.log10(np.hstack(model[:,:2])),
                    np.log10(np.hstack(model_opt_n10[:,:2])))*1000
RMSE_h_Opt_n10 = rmse_a(model[:,2], model_opt_n10[:,2])

Error_List_Noise = [['Global search [Q+IP], $\epsilon$: 2.5%', RMSE_s_GS_n2, RMSE_h_GS_n2],
              ['Global search [Q+IP], $\epsilon$: 5%', RMSE_s_GS_n5, RMSE_h_GS_n5],
              ['Global search [Q+IP], $\epsilon$: 10%', RMSE_s_GS_n10, RMSE_h_GS_n10],
              ['Optimization [Q+IP], $\epsilon$: 2.5%', RMSE_s_Opt_n2, RMSE_h_Opt_n2],
              ['Optimization [Q+IP], $\epsilon$: 5%', RMSE_s_Opt_n5, RMSE_h_Opt_n5],
              ['Optimization [Q+IP], $\epsilon$: 10%', RMSE_s_Opt_n10, RMSE_h_Opt_n10]]

RMSE = pd.DataFrame(Error_List_Noise, columns=['Method', 'RMSE $\sigma$ [mS/m]', 'RMSE $h$ [m]'])

print('Table 4 Case A.2')
print(RMSE)
