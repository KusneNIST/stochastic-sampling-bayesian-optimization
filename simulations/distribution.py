from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.stats
import attr


class Distribution(metaclass=ABCMeta):

    # @property
    # @abstractmethod
    # def p(self):
    #     pass

    @abstractmethod
    def sample(self, N=1):
        pass

    @abstractmethod
    def pdf(self, z):
        pass

    @abstractmethod
    def mean(self):
        pass


@attr.s(frozen=True)
class Normal(Distribution):

    mu = attr.ib()
    std = attr.ib()

    @property
    def _rv(self):
        return scipy.stats.norm(self.mu, self.std)

    @property
    def p(self):
        return 1

    def sample(self, N=1):
        return self._rv.rvs(size=N)

    def pdf(self, z):
        return self._rv.pdf(z)

    def mean(self):
        return self._rv.mean()


@attr.s()
class Discrete():

    support = attr.ib()
    pdfs = attr.ib()
    _pdfs = attr.ib(default=None, init=False)
    index = attr.ib(default=None, init=False)
    _p = attr.ib(default=None, init=False)

    def __attrs_post_init__(self):
        if self.support.ndim > 1:
            self._p = self.support.shape[1]
        else:
            self._p = 1
            
        self.support, self.index = np.unique(self.support, return_index=True, axis=0)
        if self.p > 1:
            self.support = list(map(tuple, self.support.tolist()))

        self.pdfs = self.pdfs[self.index]
        self._pdfs = dict(zip(self.support, self.npdfs))

    @property
    def p(self):
        return self._p

    @property
    def npdfs(self):
        "normalized"
        return self.pdfs/self.pdfs.sum()

    def sample(self, N=1):
        ret = []
        for i in range(N):

            u = scipy.stats.uniform(0, 1).rvs()
            uu = i = 0
            while uu < u:
                uu += self.npdfs[i]
                i += 1

            ret.append(self.support[i-1])

        return np.array(ret)

    def pdf(self, z):
        if self.p>1:
            if type(z) is tuple:
                z = [z]
            elif type(z) is np.ndarray:
                z = list(map(tuple, z.tolist()))

            
        ret = []
        for zz in z:
            ret.append(self._pdfs.get(zz, 0.0))

        return np.array(ret)

    def mean(self):
        return (self.npdfs*self.support).sum()


@attr.s(frozen=True)
class DiscretizedNormal(Distribution):

    mu = attr.ib()
    std = attr.ib()
    bins = attr.ib()

    @property
    def _rv(self):
        return scipy.stats.norm(self.mu, self.std)

    @property
    def p(self):
        return 1

    def pdf(self, z):
        pass


@attr.s(frozen=True)
class TruncNormal(Distribution):

    mu = attr.ib(default=0)
    std = attr.ib(default=1)
    a = attr.ib(default=-1)
    b = attr.ib(default=1)

    @property
    def p(self):
        return 1

    @property
    def _scale_min(self):
        return (self.a - self.mu)/self.std

    @property
    def _scale_max(self):
        return (self.b - self.mu)/self.std

    @property
    def _rv(self):
        return scipy.stats.truncnorm(self._scale_min, self._scale_max, self.mu, self.std)

    def pdf(self, z):
        return self._rv.pdf(z)

    def sample(self, N=1):
        return self._rv.rvs(N)

    def mean(self):
        return self.mu


@attr.s(frozen=True)
class PointMass(Distribution):

    m = attr.ib(default=0)

    @property
    def p(self):
        return 1

    def pdf(self, z):
        return np.where(z == self.m, 1, 0)

    def sample(self, N=1):
        return np.repeat(self.m, N)

    def mean(self):
        return self.m
