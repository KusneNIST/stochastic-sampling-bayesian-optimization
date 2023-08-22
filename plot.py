import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from marginal.mc import MonteCarlo

def scatterHeatmap(x, y, bins=50, weights=None,):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=weights)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower', norm=LogNorm(), aspect='auto')
    plt.colorbar()

def interval(x, fmu, fvar, alpha=.1, *args, **kwargs):
    
    if fmu.ndim > 1:
        fmu = fmu[:,0]
    if fvar.ndim > 1:
        fvar = fvar[:,0]
    if x.ndim > 1:
        x = x[:,0]
    
    l, = plt.plot(x, fmu, *args, **kwargs)
    plt.fill_between(x,
                     fmu-np.sqrt(fvar)*2,
                     fmu+np.sqrt(fvar)*2,
                     alpha=alpha, color=l.get_c()
    )

def marginals(x, f, xsamp, ysamp, m, px = [], d1 = 5, d2= 1, fs=16,
              f_range_expand=.2, marginal_method=MonteCarlo,
              marginal_kwargs = {}):

    axes = []
    
    ax = plt.subplot2grid((d1+d2, d1+d2), (0, 0), colspan=d1, rowspan=d1)
    axes.append(ax)

    plt.plot(x, f, 'k')
    plt.scatter(xsamp, ysamp)
    interval(x, *m.predict(x))
    plt.title('$p(f | x)$', fontsize=fs)
    plt.xticks([])
    
    ax = plt.subplot2grid((d1+d2, d1+d2), (d1, 0), colspan=d1, rowspan=d2)
    axes.append(ax)
    
    xlim = (np.inf, -np.inf)
    for ppx in px:
        plt.plot(x, ppx.pdf(x))
        xlim = (min(xlim[0], ppx.ppf(.01)), max(xlim[1], ppx.ppf(.99)))
    #plt.xlim(xlim)
    plt.xlabel('$x$', fontsize=fs)
    plt.ylabel('$p(x | \Theta)$', fontsize=fs)
        
    ax = plt.subplot2grid((d1+d2, d1+d2), (0, d1), colspan=d2, rowspan=d1)
    axes.append(ax)
    ax.yaxis.set_label_position("right")
    
    r = f.max() - f.min()
    z = np.linspace(f.min() - f_range_expand*r, f.max()+f_range_expand*r, 300)
    for ppx in px:
        mcm = marginal_method(m, ppx, **marginal_kwargs)
        plt.plot(mcm.pdf(z), z)

    plt.title('$p(f | \Theta)$', fontsize=fs)
