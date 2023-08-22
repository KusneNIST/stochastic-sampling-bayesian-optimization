import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import os
import attr

@attr.s
class Aggregate():
    """An aggregate of a simulation output"""

    @staticmethod
    def from_objective(path, target, ind, step=1, *args, **kwargs):
        aggs = {}

        # for each aqusition
        for aquisition in sorted(os.listdir(path)):
            if os.path.isdir(os.path.join(path, aquisition)):
                aggs[aquisition] = {}
                # for each config
                for config in sorted(os.listdir(os.path.join(path, aquisition))):
                    if os.path.isdir(os.path.join(path, aquisition, config)):
                        agg = Aggregate.build(
                            os.path.join(path, aquisition, config),
                            target, ind, step, *args, **kwargs
                        )

                        l = '{}-{}'.format(aquisition, config, )

                        aggs[aquisition][l] = agg

        return aggs


    @staticmethod
    def build(path, target, ind, step=1, cumsum=False, cummin=False,
              *args, **kwargs):
        dirs = os.listdir(path)

        ret = {}
        for d in dirs:
            p = os.path.join(path, d)
            if not os.path.isdir(p):
                continue
            
            if target in os.listdir(p):
                ret[d] = pd.read_csv(os.path.join(path, d, target), header=None)

        k = len(ret.keys())
        n = max([d.values[::step, :].shape[0] for _, d in ret.items()])
        p = len(ind)
        m = np.zeros((k, n, p))
        m[:] = np.nan

        for i, (name, d) in enumerate(ret.items()):
            dd = d.values

            # TODO: change to average w/in a step size

            # change to cumulative if necessary
            #dd = dd.values
            if cumsum:
                dd = np.cumsum(dd, 0)
            elif cummin:
                dd = np.array([np.min(dd[:i+1, :], 0) for i in range(dd.shape[0])])

            dd = dd[::step, ind]
            #dd = dd.iloc[:-1, :]

            m[i, :dd.shape[0], :] = dd

        return Aggregate(m)

    frame = attr.ib()

    @property
    def k(self):
        "number of runs in this aggregate"
        return self.frame.shape[0]

    @property
    def n(self):
        "length of aggregate"
        return self.frame.shape[1]

    @property
    def p(self):
        "number of parameters in aggregate"
        return self.frame.shape[2]

    @property
    def mean(self):
        return np.nanmean(self.frame, 0)

    @property
    def median(self):
        return np.nanmedian(self.frame, 0)

    @property
    def std(self):
        return np.nanstd(self.frame, 0)

    @property
    def min(self):
        return np.nanmin(self.frame, 0)

    @property
    def max(self):
        return np.nanmax(self.frame, 0)

    def percentile(self, q):
        return np.nanpercentile(self.frame, q, 0)
        
       

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('target')
    parser.add_argument('ind', nargs='+', type=int)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--log-space', action='store_true',dest='logspace')
    parser.add_argument('--cumsum', action='store_true')
    parser.add_argument('--cummin', action='store_true')
    parser.add_argument('-E', action='store_true', dest='errorbar')
    parser.add_argument('-I', action='store_true', dest='individual')
    parser.add_argument('--min', default = -np.inf, type=float)

    args = parser.parse_args()
    n = 0
    p = 0

    aggs = {}
    # for each aqusition
    for aquisition in sorted(os.listdir(args.path)):
        if os.path.isdir(os.path.join(args.path, aquisition)):
            aggs[aquisition] = {}
            # for each config
            for config in sorted(os.listdir(os.path.join(args.path, aquisition))):
                if os.path.isdir(os.path.join(args.path, aquisition, config)):
                    try:
                        agg = Aggregate.build(
                            os.path.join(args.path, aquisition, config),
                            args.target, args.ind, args.step, args.cumsum, args.cummin
                        )
                    except:
                        continue

                    n = max(n, agg.n)
                    p = max(p, agg.p)
                    l = '{}-{}'.format(aquisition, config, )

                    aggs[aquisition][l] = agg

    A = len(aggs.keys())
    if args.individual:
        plt.figure(figsize=(4*A, 4*p))

    for i, k in enumerate(aggs.keys()):
        for j, (l, agg) in enumerate(aggs[k].items()):
            if args.individual:
                for pp in range(p):
                    plt.subplot(A, p, i*p + pp +1)
                    if pp == 0:
                        plt.title(k)

                    x = agg.frame[:,:,pp].T
                    x[x<args.min] = args.min
                    plt.plot(x, color='C{}'.format(j), alpha=.6)

                    # plt.plot(agg.frame[:,:,pp].T, color='C{}'.format(j), alpha=.6)

                    if args.logspace:
                        plt.semilogy()
            else:
                std = agg.std
                plt.errorbar(np.arange(agg.n), agg.mean, yerr=std, label=l)

    if args.logspace:
        plt.semilogy()

    if not args.individual:
        plt.legend()

    plt.tight_layout()
        
    plt.savefig(
        os.path.join(
            args.path,
            '{}-{}.pdf'.format(
                args.target, ','.join(map(str, args.ind))
            )),
        bbox_inches='tight'
    )


                    

    # p = AggregatePlot.build(args.path, args.target, args.ind,
    # errorbar=args.errorbar, individual=args.individual)
    # 
    # n = '{}-{}'.format(os.path.split(args.path)[-1], args.target)
    # pth = os.path.join(*os.path.split(args.path)[:-1])
    # 
    # p()
    # plt.savefig(os.path.join(pth, '{}.pdf'.format(n)))
    
