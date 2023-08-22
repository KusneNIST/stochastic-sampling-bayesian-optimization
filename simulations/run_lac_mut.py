import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import attr
import logging
import sys
import scipy.stats
import GPy
from itertools import combinations, product

from objective import Objective
from parameter import Space, Builder
from modeler import Modeler
from distribution import Discrete, PointMass, Distribution
from optimizer import Optimizer
from plot import AquisitionPlot, ModelPlot, RegretPlot
import aquisition
import optimizer

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

tlogger = logging.getLogger('GP')
tlogger.setLevel(logging.ERROR)

tlogger = logging.getLogger('matplotlib')
tlogger.setLevel(logging.ERROR)

# Precursors

@attr.s
class Choice(Distribution):
    
    p = attr.ib(default=[.5, .5])
    
    @property
    def c(self):
        "number of choices"
        return len(self.p)
    
    def sample(self, N=1):
        return np.random.choice(np.arange(self.c), N, p=self.p)
    
    def pdf(self, z):
        return [self.p[zz] for zz in z]
    
    def mean(self):
        return np.arange(self.c)[np.argmax(self.p)]

@attr.s
class LacI(Objective):
    
    ind = attr.ib(default=None)
    _opt = attr.ib(default=None, init=False)

    def __attrs_post_init__(self, ):
        self.linear = pd.read_csv("data/lac-params-best-linear.csv", index_col=0)
        self.quad = pd.read_csv("data/lac-params-best-quad.csv", index_col=0)

        # sort the columns
        self.quad = self.quad.sort_index(ascending=False)
        self.quad.columns = self.quad.columns.astype(float)
        self.quad = self.quad.sort_index(1, ascending=False)
    
    def eval(self, x):
        x = np.array(x)

        if x.ndim == 1:
            x = x[None, :]
            
        return np.apply_along_axis(self._eval, 1, x).ravel()
        #return self._eval(x)
    
    def _eval(self, x):
        x = self.buildX(x)
        
        if x.ndim == 1:
            x = x[None, :]
        
        return np.dot(x, self.linear.values[:,0]) + np.dot(x[:, 1:], np.dot(self.quad, x[:, 1:].T))
    
    def buildX(self, z):
        if z.ndim > 1 and z.shape[0] > 1:
            return np.apply_along_axis(self.buildX, 1, z)
        elif z.ndim == 1:
            z = z[None, :]

        x = np.zeros(self.linear.shape[0])
        x[0] = 1
        
        _, tmp = np.where(z); #print(tmp)
        x[(self.ind*3)[tmp] + z[:, tmp]] = 1
        
        return x
    
    @property
    def range(self):
        return [[0, 3]] * self.d
    
    @property
    def optimum(self):
        if self._opt is None:
            x = self.sample_space()
            self._opt = x[np.argmax(self.eval(x)), :]
            
        return self._opt
    
    def sample_space(self, *args, **kwargs):
        return np.array(list(product(*[np.arange(4)]*self.d)))

    @property
    def kernel(self):
        return GPy.kern.Linear(self.d*3, ARD=True)

    @property
    def unrolled_ind(self):
        "active indices unrolled for different mutation values"
        return 1 + np.repeat(self.ind*3, 3) + np.tile(np.arange(3), self.ind.shape[0])

# Pathches
class LacIModeler(Modeler):

    def transform(self, x, reverse=False):
        z = x.copy()
        
        if reverse:
            raise ValueError("haven't done reverse yet!")

        x = self.objective.buildX(z)
        if x.ndim == 1:
            x = x[None, :]
        return x[:, self.objective.unrolled_ind]

    def input_space(self, *args, **kwargs):
        return self.transform(self.objective.input_space())

if __name__ == "__main__":

    import os
    import argparse
    import json

    parser = argparse.ArgumentParser()

    # directory from which simulation will be run
    # must contain a config file
    parser.add_argument('path')

    parser.add_argument('--ident', default=None, help='identifier')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite if exists')
    parser.add_argument('--plot', action='store_true', help='generate plots')

    aquisitions = aquisition.buildParser(
        parser.add_subparsers(title="aquisition functions")
    )

    args = parser.parse_args()

    config = json.load(
        open(
            os.path.join(args.path, "cfg.json")
        )
    )

    # lac specific params
    target = config.get("target", "crp1") # crp1/2, rnap1/2
    mutation_rates = config.get("mutation-rates", [.1])

    # general params
    iterations = config.get("iterations", 10)
    sample_size = config.get("sample-size", 1)
    model_resample = config.get('model-resample', 3)
    N = config.get('iterations', 10)
    noiseless = config.get('noiseless', True)
    hyperparam_opt = config.get("optimize-hyperparameters", True)
    
    args.sample_size = sample_size

    # get aquisition
    aq = args.aq_parse.build(args)

    # check path
    path = os.path.join(
        args.path,
        args.aquisition,
        args.aq_parse.path(args)
    )

    if args.ident:
        path = os.path.join(path, args.ident)

    odir = path

    if os.path.exists(odir) and not args.overwrite:
        logger.info('{} already exists, skipping'.format(odir))
        import sys
        sys.exit()

    # get params

    # build space
    logger.debug("building sample space")
    rnap1 = 75 - np.arange(7, 12)
    rnap2 = 75 - np.arange(31, 36)
    crp1 = 75 - np.arange(54, 59)
    crp2 = 75 - np.arange(65, 70)

    if target == "crp1":
        ind = crp1
    elif target == "crp2":
        ind = crp2
    elif target == "rnap1":
        ind = rnap1
    elif target == "rnap2":
        ind = rnap2
    else:
        raise ValueError("invalid target!")

    objective = LacI(ind.shape[0], ind)
    seeds = Space(np.arange(4))
    for i in range(1, objective.d):
        seeds = seeds * Space(np.arange(4))

    rates = Space(np.array(mutation_rates))

    sspace = seeds * rates

    b = Builder(sspace)

    def choices(nuc, mut):
        nuc = int(nuc)
        ret = np.ones(4)*mut/3
        ret[nuc] = 1-mut
        return ret

    for i in range(objective.d):
        b.register(lambda nuc, mut: Choice(choices(nuc, mut)), i=[i, -1])

    x = objective.sample_space()

    # build optimizer
    opt = Optimizer(objective, b, aq, sample_space=x,
                       sample_size=sample_size,
                       model_resample=model_resample, output=True,
                       output_dir=odir, hyperparam_opt=hyperparam_opt,
                       minimize=False)
    # patch in new modeler
    opt.modeler = LacIModeler(objective, opt.ystd==0, model_resample if hyperparam_opt else 0, objective.kernel)

    opt.modeler.noiseless = noiseless
    opt.optimize(N)
