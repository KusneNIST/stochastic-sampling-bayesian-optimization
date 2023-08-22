if __name__ == "__main__":

    import matplotlib as mpl
    mpl.use('Agg')

    import argparse
    import objective
    import aquisition
    from optimizer import Optimizer
    from parameter import Space, Builder
    from distribution import Normal
    from operator import mul
    from functools import reduce
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument('objective')
    parser.add_argument('--M', default=10, help='number of unique means', type=int)
    parser.add_argument('--S', default=5, help='number of unique standard deviations', type=int)
    parser.add_argument('--N', default=5, help='number of optimization steps', type=int)
    parser.add_argument('--T', default=100, help='number of monte-carlo steps', type=int)

    args = parser.parse_args()

    obj = objective.Objective.find_objective(args.objective)(d=2)
    aq = aquisition.MonteCarlo(T=args.T)

    mus = [Space(np.linspace(l + .1*(h-l), h-.1*(h-l), args.M)) for l, h in obj.range]
    diff = np.mean([h-l for l, h in obj.range])
    std = Space(np.logspace(np.log10(.01*diff), np.log10(.25*diff), args.S))
    spc = reduce(mul, mus) * std
    
    b = Builder(spc)
    b.register(Normal, [0, 2])
    b.register(Normal, [1, 2])
    
    opt = Optimizer(obj, b, aq,
                    model_resample=3, output=True, output_dir='figures/{}-montecarlo-t{}'.format(args.objective, args.T))
    opt.optimize(args.N)

