# Plot grid

def PlotModelCond_2lay(model, model_true, depthmax = 10, vmin=1, vmax=1000):
    # Arrays for plotting
    npos = np.shape(model)[0]
    ny = 71 # size of the grid in y direction
    y = np.linspace(0, depthmax, ny)
    grid = np.zeros((npos, ny))
    xx = np.linspace(0,npos+1,npos+1, endpoint=False)
    sig = model[:,:2].copy()
    thk = model[:,2].copy()
    thk_true = model_true[:,2].copy()

    # Conductivities to be plotted in a grid
    for i in range(npos):
        y1=0
        while y[y1] <= thk:
            grid[i, y1] = sig[i, 0]
            y1 += 1
        grid[i, y1:] = sig[i, 1]
    
    fig, ax = plt.subplots(figsize = (6,4))
    pos = ax.imshow((grid*1000).T, cmap='viridis', interpolation='none', 
                    extent=[0,npos,depthmax,0], norm = colors.LogNorm(vmin=vmin, vmax=vmax))
    plt.step(np.hstack((xx, xx[-1])), np.hstack((thk_true[0], thk_true[0:], thk_true[-2])), ':r')
    clb = fig.colorbar(pos, shrink=0.5)
    clb.set_label('Electrical conductivity [mS/m]',  )
    ax.set_ylabel('Depth [m]')
    ax.set_xlabel('Horizontal distance [m]')
    return grid