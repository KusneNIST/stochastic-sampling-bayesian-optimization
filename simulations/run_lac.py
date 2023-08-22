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


def buildParser(parser):

    from aquisition import AquisitionParse, BatchMatch, Mean, Random, SBUCB_LocalPenalty, StochasticBUCB, Independent

    # batch match

    bm = parser.add_parser('batchmatch', help='batch matching approximation')
    ap = AquisitionParse('bm', BatchMatch)
    bm.set_defaults(aq_parse=ap, aquisition='batchmatch')

    # null mean

    nm = parser.add_parser('mean', help='null mean aquisition')
    ap = AquisitionParse('mn', Mean)
    nm.set_defaults(aq_parse=ap, aquisition='mean')

    # random

    rm = parser.add_parser('random', help='random sampling aquisition')
    ap = AquisitionParse('rm', Random)
    rm.set_defaults(aq_parse=ap, aquisition='random')

    # stochastic batch

    class SBUCB_LP_PATCH(SBUCB_LocalPenalty):
        
        def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

            # compute the pdf for each samplex position
            pdf = dist[0].pdf(samplex)

            # make sure nothing has gone wrong in pdf calc
            assert pdf.ndim < 3
            if pdf.ndim == 2:
                assert pdf.shape[1] == 1
                pdf = pdf[:, 0]

            gamma = np.sum(pdf[:, None] * self.gamma, 0)

            a = 0
            for i in range(self.sample_size):
                a += np.sum(pdf * self._alpha[i] * np.power(gamma, i))

            return a

    sb = parser.add_parser('sbucb', help='stochastic batch UCB')
    ap = AquisitionParse('sb', SBUCB_LP_PATCH)
    sb.set_defaults(aq_parse=ap, aquisition='sbucb')

    # stochastic batch

    sb = parser.add_parser('sbucb.mc', help='stochastic batch UCB')
    sb.add_argument('--T', default=50, help='mc samples',
                    type=int, dest='sbold_T')
    ap = AquisitionParse('sbold', StochasticBUCB)
    sb.set_defaults(aq_parse=ap, aquisition='sbucb.old')

    # independent

    sb = parser.add_parser('indep', help='independent draws')
    ap = AquisitionParse('indep', Independent)
    sb.set_defaults(aq_parse=ap, aquisition='indep')

    return parser

@attr.s
class Enumerated(Objective):
    x = attr.ib(default=None)
    y = attr.ib(default=None)

    def _eval(self, x):
        return self.y[self.lookup(x)]

    def lookup(self, x):
        "find the index of the input value"
        if x.ndim > 1 and x.shape[0] > 1:
            return np.apply_along_axis(self.lookup, 1, x)
        elif x.ndim == 1:
            x = x[None, :]

        i = np.argmax(np.all(self.x==x, axis=1))
        return i

    @property
    def optimum(self):
        return self.x[np.argmax(self.y)]

    @property
    def range(self):
        return np.array([[x.min(), x.max()]])

    @property
    def kernel(self):
        return GPy.kern.Linear(self.d, ARD=True)

class LacOptimizer(Optimizer):

    def optimize(self, N):

        for it in range(N):
            logger.info('iteration {} start'.format(it))

            logger.info('computing aquisitions')
            self.current_aquisition = self.aquisition.compute(
                self.model, self.modeler,
                self.sample_space, self.sampler,
                self.minimize, it, self.sample_space.shape[0]
            )

            i = np.argwhere(self.current_aquisition >=
                            np.nanmax(self.current_aquisition))
            assert i.ndim == 2

            if i.ravel().shape[0] == 0:
                i = np.random.choice(
                    np.arange(self.current_aquisition.shape[0]))
                logger.debug(
                    "empty index for aquisition argmax, choosing {}".format(i))
            else:
                i = np.random.choice(i.ravel())

            hp = self.sampler.space[i, :]
            dist = self.sampler.build(i)[0]
            xsamp = dist.sample(self.sample_size)
            xsamp = xsamp.tolist()

            logger.debug("x sampled: {}".format(xsamp))

            ysamp = self.objective(xsamp) + \
                scipy.stats.norm(0, self.ystd).rvs(size=self.sample_size)

            # dont let the sampler "cheat" by getting values outside
            # the objective range or noise going below optimum
            # optimum = self.objective.eval(self.objective.optimum)
            # if self.minimize:
            #     ysamp[ysamp < optimum] = optimum
            # else:
            #     ysamp[ysamp > optimum] = optimum

            logger.info('selected pi({})'.format(str(hp)))
            logger.debug("received {} with f value {}".format(
                ", ".join(
                    ['[{}]'.format(
                        ', '.join(map(lambda s: '{:.4f}'.format(s), xx))) for xx in xsamp]
                ),
                ", ".join(map(lambda s: '{:.4f}'.format(s), ysamp)
                          )))

            self.a.append(hp)
            self.x.extend(xsamp)
            self.y.extend(ysamp)
            self.aq.append(self.current_aquisition)

            self.prious_model = self.model
            self.model = self.modeler.build(self.x, self.y)

            if self.output:
                self.make_output(it)

            logger.debug('total regret: {}'.format(self.regret().sum()))
            logger.info('iteration completed')


def buildSpace(ind, mutrate, linear, quad, R=3, ):
    
    P = linear.shape[0]
    N = ind.shape[0]
    M = len(mutrate)
    
    def f(x):
        return np.dot(linear.values[:,0], x) + np.dot(x[1:], np.dot(quad, x[1:]))
    
    sample_space = np.array(list(combinations(np.arange(N), R)))
    S = sample_space.shape[0]
    
    # add mutation options
    sample_space = np.repeat(sample_space, M, axis=0)
    sample_space = np.column_stack((sample_space, np.tile(mutrate, S)))
    
    fit = []
    prob = []
    xx = []
    ss = sample_space[:, :-1].astype(int)
    for i in range(sample_space.shape[0]):
        
        mr = sample_space[i, -1]
        
        fit.append([])
        prob.append([])
        xx.append([])

        # all possible mutations
        for perm in product(*[np.arange(4)]*R):
            
            x = np.zeros(P)
            x[0] = 1

            # these are the non-zero indicies
            tmp, = np.where(perm)

            perm = np.array(perm)

            x[(ind*3)[ss[i, tmp]] + perm[tmp]] = 1
            
            fit[-1].append(f(x))
            xx[-1].append(x)
            prob[-1].append(scipy.stats.binom.pmf(x[1:].sum(), N, mr))
    
    fit = np.array(fit)
    x = np.array(xx)
    prob = np.array(prob)
    
    return x, sample_space, prob, fit, f

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

    aquisitions = buildParser(
        parser.add_subparsers(title="aquisition functions")
    )

    args = parser.parse_args()

    config = json.load(
        open(
            os.path.join(args.path, "cfg.json")
        )
    )

    # lac specific params
    crp = config.get("crp", False)
    rnap = config.get("rnap", False)
    mutation_rates = config.get("mutation-rates", [.1])
    positions = config.get("positions", 3)

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

    index = np.array([])
    if crp:
        index = np.concatenate((index, crp1, crp2))
    if rnap:
        index = np.concatenate((index, rnap1, rnap2))
    index = index.astype(int)

    # target = None
    # if crp and rnap:
    #     target="full"
    # elif crp:
    #     target="crp"
    # elif rnap:
    #     target="rnap"
    # else:
    #     raise ValueError("must choose at least crp or rnap!")

    # pth = "data/lac/{}/".format(target)

    # x = np.load(os.path.join(pth, "x.npy"))
    # sample_space = np.load(os.path.join(pth, "sample-space.npy"))
    # prob = np.load(os.path.join(pth, "prob.npy"))
    # fit = np.load(os.path.join(pth, "fit.npy"))
    # mx = np.load(os.path.join(pth, "mx.npy"))

    if not "spaces" in os.listdir(args.path):
        os.makedirs(os.path.join(args.path, "spaces"),)

    #if pth in os.listdir(os.path.join(args.path, "spaces")):
    try:
        pth = "{}-{}".format(",".join(map(str, mutation_rates)), positions)
        pth = os.path.join(args.path, "spaces", pth)

        x = np.load(os.path.join(pth, "x.npy"))
        sample_space = np.load(os.path.join(pth, "sample-space.npy"))
        prob = np.load(os.path.join(pth, "prob.npy"))
        fit = np.load(os.path.join(pth, "fit.npy"))
        mx = np.load(os.path.join(pth, "mx.npy"))

    except:
        logger.debug("issue loading from disk, creating from scratch")

        pth = "{}-{}".format(",".join(map(str, mutation_rates)), positions)
        pth = os.path.join(args.path, "spaces", pth)
        os.makedirs(pth, exist_ok=True)

        opt = "-best"
        linear = pd.read_csv("data/lac-params{}-linear.csv".format(opt), index_col=0)
        quad = pd.read_csv("data/lac-params{}-quad.csv".format(opt), index_col=0)

        x, sample_space, prob, fit, f = buildSpace(index, mutation_rates, linear, quad, R=positions, )
        mind, = np.where(np.any(np.any(x[:, :, 1:]==1, axis=0), axis=0))
        mind = mind + 1
        mx = x[:, :, mind]

        np.save(os.path.join(pth, "x.npy"), x)
        np.save(os.path.join(pth, "sample-space.npy"), sample_space)
        np.save(os.path.join(pth, "prob.npy"), prob)
        np.save(os.path.join(pth, "fit.npy"), fit)
        np.save(os.path.join(pth, "mx.npy"), mx)

    V = prob.shape[1] # number of unique non-zero variants in each distribution

    # flatten sample space for model
    mmx = mx.reshape((mx.shape[0] * mx.shape[1], mx.shape[2]))
    fft = fit.ravel()

    # build sampler
    # will use index of hyperparam space to build discrete sampler of different values
    logger.debug("build sampler")
    builder = Builder(np.arange(sample_space.shape[0])[:, None])
    def bld(i, mmx=mmx):
        support = mmx[np.arange(V) + V*i, :]
        pdfs = prob[i, :]
        return Discrete(support, pdfs)
    builder.register(bld, [0])

    # remove redundant x
    mmx, rind = np.unique(mmx, return_index=True, axis=0)
    fft = fft[rind]

    # build objective
    objective = Enumerated(mmx.shape[1], mmx, fft)

    # build optimizer
    opt = LacOptimizer(objective, builder, aq, sample_space=mmx,
                       sample_size=sample_size,
                       model_resample=model_resample, output=True,
                       output_dir=odir, hyperparam_opt=hyperparam_opt,
                       minimize=False)
    opt.modeler.noiseless = True
    opt.optimize(N)

    sys.exit()

    #opt = Optimizer(objective, sampler, aquisition, mmx, sample_size, ystd, minimize=False, model_resample)

    model = None
    modeler = None
    aq = None
    sampler = None
    minimize = False
    hpsamp = []
    xall = []
    yall = []

    # run simulation
    for i in range(iterations):
        logger.info('iteration {} start'.format(it))

        logger.info('computing aquisitions')
        current_aquisition = aq.compute(
            model, modeler,
            sample_space, sampler,
            minimize, i, sample_space.shape[0]
        )

        i = np.argwhere(current_aquisition >= np.nanmax(current_aquisition))
        assert i.ndim == 2
        i = np.random.choice(i.ravel())

        hp = sampler.space[i, :]
        xsamp = np.column_stack([d.sample(sample_size)
                                    for d in sampler.build(i)])
        xsamp = xsamp.tolist()

        ysamp = np.array([objective(xs) + scipy.stats.norm.rvs(0, ystd) for xs in xsamp])

        logger.info('selected pi({})'.format(str(hp)))
        logger.debug("received {} with f value {}".format(
            ", ".join(
                ['[{}]'.format(', '.join(map(lambda s: '{:.4f}'.format(s), xx))) for xx in xsamp]
            ),
            ", ".join(map(lambda s: '{:.4f}'.format(s), ysamp)
            )))

        hpsamp.append(hp)
        xall.extend(xsamp)
        yall.extend(ysamp)

        previous_model = model
        model = modeler.build(xall, yall)

        if self.output:
            self.make_output(it)

        logger.debug('total regret: {}'.format(self.regret().sum()))
        logger.info('iteration completed')
