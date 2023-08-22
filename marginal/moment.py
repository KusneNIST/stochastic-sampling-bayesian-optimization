import GPy
from marginal.mc import MonteCarlo
import numpy as np

class Moment():

    @staticmethod
    def _train_gp(x, y, k, n, randomizer):
        best = None

        for _ in range(n):
            m = GPy.models.GPRegression(x, y, k.copy())
            randomizer(m)
            m.optimize()

            if best is None or m.log_likelihood() > best.log_likelihood():
                best = m

        return best

    @classmethod
    def randomizer(cls, m):
        m.kern.lengthscale.randomize()


    def __init__(self, theta, m, pix_gen, ntrain=5, mc_kwargs={}):
        self.theta = theta
        self.m = m
        self.pix_gen = pix_gen

        self.approxes = []
        for th in theta:
            self.approxes.append(MonteCarlo(m, pix_gen(*th), **mc_kwargs))

        self.m_mu = Moment._train_gp(self.theta,
                                     np.array([a.mean() for a in self.approxes])[:,None],
                                     GPy.kern.RBF(self.p, ARD=True),
                                     ntrain, self.randomizer)

        self.m_std = Moment._train_gp(self.theta,
                                     #np.array([np.log10(a.var()**.5) for a in self.approxes])[:,None],
                                     np.array([a.var()**.5 for a in self.approxes])[:,None],
                                     GPy.kern.RBF(self.p, ARD=True),
                                     ntrain, self.randomizer)

                                     
        # self.m_mu = GPy.models.GPRegression(
        #     self.theta, 
        #     np.array([a.mean() for a in self.approxes])[:,None],
        #     GPy.kern.RBF(self.p, ARD=True))

        # self.m_mu.optimize()

        # self.m_std = GPy.models.GPRegression(
        #     self.theta, 
        #     np.array([np.log10(a.var()**.5) for a in self.approxes])[:,None],
        #     GPy.kern.RBF(self.p, ARD=True))

        # self.m_std.optimize()

    @property
    def p(self):
        return self.theta.ndim
