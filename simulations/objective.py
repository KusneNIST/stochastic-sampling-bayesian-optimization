from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import attr
import GPy

@attr.s
class Objective(metaclass=ABCMeta):
    d = attr.ib(default=2)

    # default noise variance to use in model
    noise_variance = 0

    @property
    @abstractmethod
    def range(self):
        pass

    @property
    @abstractmethod
    def optimum(self):
        pass

    @property
    def kernel(self):
        return GPy.kern.RBF(self.d, ARD=True)

    def eval(self, x):
        x = np.array(x)

        if x.ndim == 1:
            x = x[None, :]

        return self._eval(x)

    @abstractmethod
    def _eval(self, x):
        pass

    def __call__(self, x):
        return self.eval(x)

    @classmethod
    def find_objective(cls, o):
        for obj in cls.__subclasses__():
            if o.lower() in obj.__name__.lower():
                return obj

        return None

    def sample_space(self, N=50, rng=None):
        """generate values at which to evaluate the objective"""

        if rng is None:
            rng = self.range

        return np.column_stack([
            zz.ravel() for zz in
            np.meshgrid(*[
                np.linspace(l, h, N) for l, h in rng
            ])
        ])
 
@attr.s
class Schwefel(Objective):

    @property
    def range(self):
        return [(-500, 500)] * self.d

    @property
    def dimensions(self):
        return self.d

    def _eval(self, x):
        return 418.9829 * self.d - (x * np.sin(np.sqrt(np.abs(x)))).sum(1)

    @property
    def optimum(self):
        return [420.9687] * self.d

    @property
    def kernel(self):
        kern = None

        for i in range(self.d):
            if kern is None:
                kern = GPy.kern.RBF(1, variance=.9, lengthscale=.05)
                kern.lengthscale.set_prior(GPy.priors.Gamma(1, 1))
            else:
                temp = GPy.kern.RBF(1, variance=.9, lengthscale=.05, active_dims=[i])
                temp.lengthscale.set_prior(GPy.priors.Gamma(1, 1))
                kern = kern + temp

        return kern
    
    
@attr.s
class Ackley(Objective):
    a = attr.ib(default=20)
    b = attr.ib(default=.2)
    c = attr.ib(default=2*np.pi)


    def _eval(self, x):
        return -self.a * np.exp(-self.b * np.sqrt(1/self.d * (x**2).sum(1))) \
            - np.exp(1/self.d * np.cos(self.c * x).sum(1)) + self.a + np.exp(1)
    
    @property
    def optimum(self):
        return [0] * self.d
    
    @property
    def range(self):
        return [(-32.768, 32.768)] * self.d

    @property
    def kernel(self):
        return GPy.kern.RBF(self.d, variance=0.43607711, lengthscale=0.0972672) + \
                GPy.kern.RBF(self.d, variance=0.05482391, lengthscale=0.00329039)

class Dropwave(Objective):

    def _eval(self, x):
        return - (1 + np.cos(12 * np.sqrt((x**2).sum(1)))) / (.5 * (x**2).sum(1) + 2)

    @property
    def range(self):
        return [(-5.12, 5.12)] * self.d

    @property
    def optimum(self):
        return [0] * self.d

    @property
    def kernel(self):
        return GPy.kern.RBF(self.d, variance=1.54835186, lengthscale=0.01433737) *\
            GPy.kern.Cosine(self.d, variance=0.60735844, lengthscale=0.30507804)
    
class Griewank(Objective):

    def _eval(self, x):
        return 1/4000 * (x**2).sum(1) - np.prod(np.cos(x/np.sqrt(1 + np.arange(self.d))), 1) + 1

    @property
    def range(self):
        return [(-100, 100)] * self.d

    @property
    def optimum(self):
        return [0] * self.d

    @property
    def kernel(self):
        return GPy.kern.RBF(self.d, variance=8.10768336e+02, lengthscale=2.12955870e+00) +\
            GPy.kern.RBF(self.d, variance=1.96458744e-01, lengthscale=5.41434080e-03)

@attr.s
class Michalewicz(Objective):

    m = attr.ib(default=10)

    def _eval(self, x):
        return - (np.sin(x) * \
                  np.power(np.sin((1 + np.arange(self.d)) * np.power(x, 2) / np.pi),\
                           2*self.m)).sum(1)

    @property
    def range(self):
        return [(0, np.pi)] * self.d

    @property
    def optimum(self):
        if self.d > 10:
            raise ValueError("Unknown optimum!")

        # from Vanaret et al.,
        # "Certified Global Minima for a Benchmark of Difficult Optimization Problems"
        return [2.202906, 1.570796, 1.284992, 1.923058, 1.720470,
                1.570796, 1.454414, 1.756087, 1.655717, 1.570796][:self.d]

    @property
    def kernel(self):
        return GPy.kern.RBF(self.d, variance=0.36189206, lengthscale=0.03873184)

class Rastrigin(Objective):

    def _eval(self, x):
        return 10 * self.d + (x**2 - 10 * np.cos(2 * np.pi * x)).sum(1)

    @property
    def range(self):
        return [(-5.12, 5.12)]*self.d

    @property
    def optimum(self):
        return [0] * self.d

    @property
    def kernel(self):

        kern = None

        for i in range(self.d):
            if kern is None:
                kern = GPy.kern.RBF(1, variance=.5, lengthscale=.05)
                kern.lengthscale.set_prior(GPy.priors.Gamma(.5, 5))
                kern.variance.set_prior(GPy.priors.Gamma(.5, 1))
            else:
                temp = GPy.kern.RBF(1, variance=.5, lengthscale=.05, active_dims=[i])
                temp.lengthscale.set_prior(GPy.priors.Gamma(.5, 5))
                temp.variance.set_prior(GPy.priors.Gamma(.5, 1))
                kern = kern + temp

        return kern

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('objectives', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if len(args.objectives) == 0:
        objectives = Objective.__subclasses__()
    else:
        objectives = []
        targets = [o.lower() for o in args.objectives]
        for obj in Objective.__subclasses__():
            if obj.__name__.lower() in targets:
                objectives.append(obj)

    n = int(np.ceil(np.sqrt(len(objectives))))
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(2*n, 2*n))

    used = []
    for ax, obj in zip(axes.ravel().tolist(), objectives):
        print(obj.__name__)
        used.append(ax)
        obj = obj(d=2)

        N = 100
        x, y = np.meshgrid(*[np.linspace(*r, N) for r in obj.range])

        e = np.zeros((N, N))

        for i in range(N):
            for j in range(N):
                #e[N-i-1, j] = obj.eval(np.array([x[i, j], y[i, j]]))
                e[i, j] = obj.eval(np.array([x[i, j], y[i, j]]))

        ext = np.array(obj.range).ravel()
        ax.imshow(e, extent=ext, aspect='auto', origin='lower')
        ax.plot(*obj.optimum, marker='x', color='r')
        #ax.colorbar()
        ax.set_title(obj.__class__.__name__)

    for ax in axes.ravel().tolist():
        if not ax in used:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/objectives.pdf')
        
