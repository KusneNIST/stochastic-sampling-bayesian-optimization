from abc import ABCMeta, abstractmethod
from itertools import product
import attr
import numpy as np
from functools import reduce

class SpaceLike(metaclass=ABCMeta):

#     @abstractmethod
#     def sample(self):
#         pass

    def __getitem__(self, i):
        return self.values[i]


    @property
    def n(self):
        return self.values.shape[0]

    @property
    @abstractmethod
    def k(self):
        """the dimensionality of the space"""
        pass

    @property
    @abstractmethod
    def shape(self):
        """the shape of each dimension in the space"""
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __mul__(self, o):
        return Cartesian.product(self, o)

@attr.s(frozen=True)
class Space(SpaceLike):

    values = attr.ib()

    def __len__(self):
        return len(self.values)

    @property
    def k(self):
        return 1

    @property
    def shape(self):
        return (len(self.values), )

    def __contains__(self, i):
        return False
        
@attr.s(frozen=True)
class Cartesian(SpaceLike):

    @staticmethod
    def product(s1, s2):

        if isinstance(s1, Cartesian):
            s1 = [s for s in s1.spaces]
        else:
            s1 = [s1]

        if isinstance(s2, Cartesian):
            s2 = [s for s in s2.spaces]
        else:
            s2 = [s2]
            
        return Cartesian(s1 + s2)

    spaces = attr.ib()

    @property
    def values(self):
        v = list(product(*[s.values for s in self.spaces]))
        return np.array(v)

    @property
    def k(self):
        return sum([s.k for s in self.spaces])

    @property
    def shape(self):
        return reduce(lambda x, y: (x+y), [s.shape for s in self.spaces])

    def __len__(self):
        return np.product([len(s) for s in self.spaces])

@attr.s
class Builder():

    space = attr.ib()
    _registers = attr.ib(init=False, factory=list)

    def register(self, cls, i=0):
        self._registers.append((cls, i))

    def build(self, i=0):

        v = self.space[i]

        ret = []
        for cls, ind in self._registers:
            ret.append(cls(*v[ind]))

        return ret
