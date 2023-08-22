import numpy as np
import scipy

class GaussianApprox():
    
    def __init__(self, m, px):
        self.m = m
        self.px = px
        self._mu, self._var = m.predict(np.array([[self.px.mean()]]))


    def pdf(self, z):
        return scipy.stats.norm(self.mean(), self.var()**.5).pdf(z)
        
    def mean(self,):
        return self._mu[0, 0]
    
    def var(self):
        
        k_inv = np.linalg.inv(self.m.kern.K(self.m.X))
        kmx = self.m.kern.K(self.m.X, np.array([[self.px.mean()]]))
        diff = self.m.X - self.px.mean()
        
        ddvar_dxdx = -2 / (self.m.kern.lengthscale**2) * np.dot((diff*kmx).T, np.dot(k_inv, (diff*kmx))) \
                    + np.dot((diff*diff*kmx).T, k_inv).dot(kmx) \
                    + 2 / self.m.kern.lengthscale * np.dot(kmx.T, np.dot(k_inv, kmx))
        t2 = .5 * np.trace(ddvar_dxdx*self.px.var())
        
        dvar_dx = 1/self.m.kern.lengthscale * (diff*kmx).T.dot(k_inv).dot(self.m.Y)
        t3 = self.px.var() * np.dot(dvar_dx.T, dvar_dx)
        
        v = self._var + t2 + t3
        return max(v[0,0], 1e-9)
