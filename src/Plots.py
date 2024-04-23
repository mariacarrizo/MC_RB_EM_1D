"""Model viewer functions."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

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

    if kwargs.pop('colorBar', True):
        cb = pg.viewer.mpl.createColorBar(p, cMin=cMin, cMax=cMax, nLevs=5)
#    cb = plt.colorbar(p, orientation='horizontal',aspect=50,pad=0.1)
        if 'cticks' in kwargs:
            xt = np.unique(np.clip(kwargs['cticks'], cMin, cMax))
            cb.set_ticks(xt)
            cb.set_ticklabels([str(xti) for xti in xt])
        if 'label' in kwargs:
            cb.set_label(kwargs['label'])

    plt.draw()
    return ax  # maybe return cb as well?