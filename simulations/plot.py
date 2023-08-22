from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import attr
import os
import glob
import numpy as np
from aquisition import BatchMatch

@attr.s
class Plot(metaclass=ABCMeta):
    _figure_name = ''

    @classmethod
    def curried(cls, **kwargs):
        def cur(*args, **kw):
            kw.update(kwargs)
            return cls(*args, **kw)

        class cur(cls):

            def __init__(self, *args, **kw):
                kw.update(kwargs)
                super(cur, self).__init__(*args, **kw)

        return cur

    optimizer = attr.ib()

    @abstractmethod
    def plot(self, axes, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def shape(self):
        """the shape of the plot, e.g. axes needed"""
        pass

@attr.s
class AquisitionPlot(Plot):

    _figure_name = 'aquisition'

    @property
    def shape(self):
        shape = self.optimizer.sampler.space.shape

        if len(shape) <= 2:
            return (1, 1)
        elif len(shape) == 3:
            return (1, shape[2])
        elif len(shape) == 4:
            return (shape[3], shape[2])

        raise ValueError('cannot handle dimension {} > 4!'.format(len(shape)))

    def plot(self, axes, iteration):
        aq = self.optimizer.alpha
        space = self.optimizer.sampler.space
        shape = space.shape

        if aq.ndim == 2:
            ext = np.ravel([[x.values.min(), x.values.max()] for x in space.spaces[:2]])

            ax = axes[0]
            ax.imshow(
                aq.T, extent = ext,
                vmin=aq.min(), vmax=aq.max(),
                origin='lower',
                aspect='auto'
            )

        elif aq.ndim == 3:

            ext = np.ravel([[x.values.min(), x.values.max()] for x in space.spaces[:2]])

            K = shape[2]
            for k in range(K):
                ax = axes[k]
                ax.imshow(
                    aq[:,:,k].T, extent = ext,
                    vmin=aq.min(), vmax=aq.max(),
                    origin='lower'
                )
                ax.set_title('{:.3}'.format(space.spaces[2].values[k]))

@attr.s
class ModelPlot(Plot):

    _figure_name = 'model'

    n = attr.ib(default=5)

    @property
    def shape(self):
        d = self.optimizer.objective.d
        if d == 1:
            return (2, 1)
        if d == 2:
            return (1, 2)
        elif d == 3:
            return (1, self.n)
        elif d == 4:
            return (self.n, self.n)

        raise ValueError('cannot handle dimension {} > 4!'.format(d))
            
    def plot(self, axes, iteration):

        model = self.optimizer.previous_model
        if model is None:
            return
        
        modeler = self.optimizer.modeler
        
        x = modeler.input_space()

        xs = model.X
        xs = self.optimizer.modeler.transform(xs, reverse=True)

        ys = model.Y
        ys = ys*np.std(self.optimizer.y) + np.mean(self.optimizer.y)

        d = self.optimizer.objective.d

        if d == 1:
            xs = xs[:, 0]
            ys = ys[:, 0]

            lims = (
                self.optimizer.sampler.space.values[:,0].min(),
                self.optimizer.sampler.space.values[:,0].max(),
            )
            predx = np.linspace(*lims, 500)[:,None]
            pmu, pvar = model.predict(
                self.optimizer.modeler.transform(predx)
            )

            ax1, ax2 = axes
            ax1.plot(predx, self.optimizer.objective.eval(predx), label='objective'); 
            #ax1.scatter(self.optimizer.x, self.optimizer.y); 
            ax1.scatter(xs, ys); 
            interval(pmu, pvar, predx[:,0], .4,
                     np.mean(self.optimizer.y), np.std(self.optimizer.y), ax=ax1);
            ax1.axvline(self.optimizer.objective.optimum, c='k', alpha=.3)

            ax1t = ax1.twinx()
            alpha = self.optimizer.aquisition.alpha( pmu, pvar,
                                                     len(self.optimizer.x),
                                                     self.optimizer.sampler.space.n,
                                                     self.optimizer.minimize)
            ax1t.plot(predx[:,0], alpha, c='C3', label='aquisition')
            ar = np.nanmax(alpha) - np.nanmin(alpha)
            ax1t.set_xlim(*self.optimizer.objective.range[0])
            ax1t.set_ylim(alpha.min() - .1*ar, alpha.max() + 1*ar)
            ax1.set_xlim(*lims)
            ax1t.set_yticks([])
            ax1.set_xticks([])

            p = AquisitionPlot(self.optimizer)
            p.plot([ax2], 50)

        if x.shape[1] == 2:
            ax = axes[0]

            ext = (x[:,0].min(), x[:,0].max(), x[:,1].min(), x[:,1].max())

            predx = x
            ys = ys[:, 0]

            mu, var = model.predict(modeler.transform(predx))

            r = int(np.sqrt(predx.shape[0]))

            ax.imshow(mu.reshape((r, r)), extent = ext, origin='lower')

            ax.scatter(
                xs[:, 0], xs[:, 1],
                alpha=.6, c=ys, edgecolor='k', #cmap='plasma'
            )

            if isinstance(self.optimizer.aquisition, BatchMatch):
                aq = self.optimizer.aquisition

                if not aq.xselect is None:
                    xs = self.optimizer.modeler.transform(np.array(aq.xselect), reverse=True)
                    ax.scatter(
                        xs[:,0], xs[:, 1], marker='x', c='r'
                    )

            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])

            ax = axes[1]
            aq = self.optimizer.aquisition.alpha(mu, var, iteration, x.shape[1])
            # std = np.sqrt(var)

            # if self.optimizer.minimize:
            #     aq = -mu + self.optimizer.aquisition.beta(iteration, x.ndim) * std
            # else:
            #     aq = mu + self.optimizer.aquisition.beta(iteration, x.ndim) * std

            ax.imshow(aq.reshape((r, r)), extent = ext, origin='lower')
            


@attr.s
class RegretPlot(Plot):

    _figure_name = 'regret'
    n = attr.ib(default=10)

    @property
    def shape(self):
        return (1, 1)
           
    def plot(self, axes, iteration):

        ax = axes[0]
        ax.plot(np.arange(len(self.optimizer.x)), self.optimizer.regret())
        ax.set_xlim(0, self.n-1)

@attr.s(frozen=True)
class AggregatePlot():

    @staticmethod
    def build(path, target, *args, **kwargs):
        dirs = glob.glob(path)

        ret = {}
        for d in dirs:
            if not os.path.isdir(d):
                continue
            
            if target in os.listdir(d):
                n = os.path.split(d)[-1]
                ret[d] = pd.read_csv(os.path.join(d, target))

        return AggregatePlot(ret, *args, **kwargs)

    data = attr.ib()
    ind = attr.ib()
    errorbar = attr.ib(default=False)
    individual = attr.ib(default=False)
    step = attr.ib(default=1)

    @property
    def k(self):
        return len(self.data.keys())

    @property
    def p(self):
        return len(self.ind)

    @property
    def n(self):
        return int(max([d.shape[0] for _, d in self.data.items()])/self.step)

    @property
    def frame(self):
        m = np.zeros((self.k, self.n, self.p))
        m[:] = np.nan

        for i, (_, d) in enumerate(self.data.items()):
            dd = d.iloc[::self.step, self.ind]
            dd = dd.iloc[:-1, :]

            m[i, :dd.shape[0], :] = dd

        return m

    @property
    def mean(self):
        return np.nanmean(self.frame, 0)

    @property
    def std(self):
        return np.nanstd(self.frame, 0)

    def __call__(self):

        if self.individual:
            for n, d in self.data.items():
                for pi, i in enumerate(self.ind):
                    plt.subplot(1, self.p, pi+1)
                    plt.plot(d.iloc[:, i], label=n)
            plt.subplot(1, self.p, 1)
            plt.legend()

        if self.errorbar:

            for i in range(self.p):
                plt.errorbar(np.arange(self.n), self.mean[:,i], yerr=2*self.std[:,i])

def interval(mu, var, x = None, alpha=.4, mu_transform=0, std_transform=1, ax=None):

    if ax is None:
        ax = plt.gca()

    if x is None:
        x = np.linspace(0, 1, mu.shape[0])

    if mu.ndim > 1:
        mu = mu[:, 0]

    if var.ndim > 1:
        var = var[:, 0]

    # undo a transform
    mu = std_transform * mu + mu_transform
    var = std_transform**2 * var
    

    std = np.sqrt(var)

    ax.plot(x, mu)
    ax.fill_between(x, mu - 1.98*std, mu + 1.98*std, alpha=alpha)
