import scipy
import numpy as np

class Numerical():
    
    def __init__(self, m, px, T=100, alpha=.01):
        self.m = m
        self.px = px
        self.T = T
        self.alpha = alpha

        self.z = np.linspace(*self.limits, self.T)
        self.pfx = scipy.stats.norm(*self.m.predict(self.z[:, None]))
        self.pxz = np.array([ppx.pdf(self.z) for ppx in self.px])

    @property
    def k(self):
        return len(self.px)

    @property
    def limits(self):
        low = np.inf
        hi = -np.inf

        for ppx in self.px:
            l = ppx.ppf(self.alpha/2)
            h = ppx.ppf(1-self.alpha/2)

            low = min(low, l)
            hi = max(hi, h)

        return (low, hi)
        
    def pdf(self, z):
        return self.pxz.dot(self.pfx.pdf(z)).T/self.T
    
    def mean(self):
        return self.pxz.dot(self.pfx.mean())/self.T

    def var(self):
        return self.pxz.dot(self.pfx.var())/self.T
