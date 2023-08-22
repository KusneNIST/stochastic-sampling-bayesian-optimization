import sys
import logging

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

if __name__ == "__main__":

    import os
    import argparse
    import json
    import scipy.stats
    from operator import mul
    from functools import reduce
    import numpy as np

    import optimizer
    import objective
    import aquisition
    from parameter import Space, Builder
    from distribution import Discrete, PointMass
    from plot import AquisitionPlot, ModelPlot, RegretPlot

    parser = argparse.ArgumentParser()

    # directory from which simulation will be run
    # must contain a config file
    parser.add_argument('path')

    parser.add_argument('--ident', default=None, help='identifier')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite if exists')
    parser.add_argument('--plot', action='store_true', help='generate plots')

    parser.add_argument('objective')
    aquisitions = aquisition.buildParser(
        parser.add_subparsers(title="aquisition functions")
    )

    args = parser.parse_args()

    config = json.load(
        open(
            os.path.join(args.path, "cfg.json")
        )
    )

    # general config options
    D = config.get('dimensions', 1)
    sample_size = config.get('sample-size', 1)
    model_resample = config.get('model-resample', 3)
    N = config.get('iterations', 10)
    M = config.get('means', 10)
    S = config.get('stds', 5)
    Nspace = config.get('nspace', 100)
    noiseless = config.get('noiseless', True)

    hyperparam_opt = config.get("optimize-hyperparameters", True)

    args.sample_size = sample_size

    obj = objective.Objective.find_objective(args.objective)(d=D)
    aq = args.aq_parse.build(args)

    # build distributions
    stdmin = config.get('stdmin', .001)
    stdmax = config.get('stdmax', .1)

    # TODO: add config for exact vs stochastic sampling
    dist = config.get("distribution", "normal")

    mus = [Space(np.linspace(l, h, M)) for l, h in obj.range]

    low = min([l for l, h in obj.range])
    hi = max([h for l, h in obj.range])
    d = hi-low
    u, l = stdmax, stdmin
    std = Space(np.logspace(np.log10(l*d), np.log10(u*d), S))

    sample_space = obj.sample_space(Nspace)

    if dist == "normal":
        spc = reduce(mul, mus) * std

        b = Builder(spc)
        # register trunc normal for each input dim, last column is std
        for i in range(D):
            # bld = lambda m, s: TruncNormal(m, s, *obj.range[i])
            # bld = lambda m, s: Discrete(sample_space[:, i], scipy.stats.norm(m, s).pdf(sample_space[:, i]))
            def bld(m, s):
                return Discrete(sample_space[:, i], scipy.stats.norm(m, s).pdf(sample_space[:, i]))
            b.register(bld, [i, D])

    elif dist == "uniform":
        spc = reduce(mul, mus)

        b = Builder(spc)
        for i in range(D):
            def bld(m, s):
                return Discrete(sample_space[:, i], scipy.stats.discrete(m-s/2, s).pdf(sample_space[:, i]))
            b.register(bld, [i, D])

    elif dist == "exact":
        spc = reduce(mul, mus)

        b = Builder(spc)
        for i in range(D):
            b.register(PointMass, [i])
    else:
        raise ValueError("no distribution selected!")

    logger.info("Sampler space has {} hyperparameter sets".format(len(spc)))
    logger.info("Sample space has size {}".format(sample_space.shape[0]))

    path = os.path.join(
        args.path,
        args.objective,
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

    plots = []
    if args.plot:
        plots = [
            AquisitionPlot,
            ModelPlot,
            RegretPlot.curried(n=N*sample_size)
        ]

    # for hyp in range(len(spc)):

    #     pi = b.build(hyp)

    #     print(np.product([pi[i].pdf(sample_space[:, i]) for i in range(D)], 0).sum())

    # import sys
    # sys.exit()

    opt = optimizer.Optimizer(obj, b, aq,
                              sample_space=sample_space,
                              sample_size=sample_size,
                              model_resample=model_resample,
                              output=True,
                              output_dir=odir,
                              hyperparam_opt=hyperparam_opt,
                              plots=plots,
                              )
    opt.modeler.noiseless = True
    opt.optimize(N)
