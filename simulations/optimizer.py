import matplotlib
matplotlib.use('Agg')

from abc import ABCMeta, abstractmethod
import numpy as np
import scipy
import attr
import logging
import modeler
import os
import sys
import pandas as pd

logger = logging.getLogger(__name__)


@attr.s
class Optimizer():

    objective = attr.ib()
    sampler = attr.ib()
    aquisition = attr.ib()
    sample_space = attr.ib()
    sample_size = attr.ib(default=1)
    ystd = attr.ib(default=0)
    minimize = attr.ib(default=True)
    model_resample = attr.ib(default=1)
    output = attr.ib(default=False)
    output_dir = attr.ib(default='.')
    plots = attr.ib(factory=list)
    hyperparam_opt = attr.ib(default=True)

    x = attr.ib(factory=list, init=False)
    y = attr.ib(factory=list, init=False)
    a = attr.ib(factory=list, init=False)
    theta_ind = attr.ib(factory=list, init=False)
    aq = attr.ib(factory=list, init=False)
    current_aquisition = attr.ib(default=None, init=False)
    model = attr.ib(default=None, init=False)
    previous_model = attr.ib(default=None, init=False)
    modeler = attr.ib(default=None, init=False)

    def __attrs_post_init__(self):
        if self.output:
            os.makedirs(self.output_dir, exist_ok=True)
            for p in self.plots:
                os.makedirs(os.path.join(self.output_dir,
                                         'figures', p._figure_name), exist_ok=True)

        self.modeler = modeler.Modeler(
            self.objective, self.ystd == 0,
            self.model_resample if self.hyperparam_opt else 0, self.objective.kernel)

    @property
    def alpha(self):
        """view of current aquisition values matching the shape of hyperparameter space"""
        if not self.current_aquisition is None:
            return self.current_aquisition.reshape(self.sampler.space.shape)

        return None

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
            xsamp = np.column_stack([d.sample(self.sample_size)
                                     for d in self.sampler.build(i)])
            xsamp = xsamp.tolist()

            logger.debug("x sampled: {}".format(xsamp))

            ysamp = self.objective.eval(xsamp) + \
                scipy.stats.norm(0, self.ystd).rvs(size=self.sample_size)

            # dont let the sampler "cheat" by getting values outside
            # the objective range or noise going below optimum
            optimum = self.objective.eval(self.objective.optimum)
            if self.minimize:
                ysamp[ysamp < optimum] = optimum
            else:
                ysamp[ysamp > optimum] = optimum

            logger.info('selected pi({})'.format(str(hp)))
            logger.debug("received {} with f value {}".format(
                ", ".join(
                    ['[{}]'.format(
                        ', '.join(map(lambda s: '{:.4f}'.format(s), xx))) for xx in xsamp]
                ),
                ", ".join(map(lambda s: '{:.4f}'.format(s), ysamp)
                          )))

            # save iteration
            self.a.append(hp)
            self.x.extend(xsamp)
            self.y.extend(ysamp)
            self.aq.append(self.current_aquisition)
            self.theta_ind.append(i)

            self.previous_model = self.model
            self.model = self.modeler.build(self.x, self.y)

            if self.output:
                self.make_output(it)

            logger.debug('total regret: {}'.format(self.regret().sum()))
            logger.info('iteration completed')

    def make_output(self, it):

        # do plotting
        import matplotlib.pyplot as plt
        for ip in self.plots:
            try:
                p = ip(self)

                fig, axes = plt.subplots(*p.shape,
                                         figsize=(8*p.shape[1], 4*p.shape[0]),
                                         squeeze=False)
                axes = axes.ravel()

                p.plot(axes, it)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir,
                                 'figures',
                                 p._figure_name,
                                 '{}_{}.pdf'.format(p._figure_name, it)
                                 ),
                    bbox_inches='tight'
                )
                plt.close()

            except Exception as e:
                print(e)
                continue

        # save output files
        for f, v in [
                ('hyperparams', self.a),
                ('x', self.x),
                ('y', self.y),
                #('aquisition', self.aq),
                ('regret', self.regret()),
                ('regret-theta', self.regret(True)),
        ]:
            pd.DataFrame(v).to_csv(
                os.path.join(self.output_dir, '{}.csv'.format(f)),
                index=False, header=False)

    def regret(self, theta=False):
        if theta:
            ret = []
            for i in self.theta_ind:
                dists = self.sampler.build(i)
                pdf = np.product(np.column_stack([
                        d.pdf(self.sample_space[:,j]) for j, d in enumerate(self.sampler.build(i))]),
                                 axis=1)

                assert pdf.shape[0] == self.sample_space.shape[0]

                f = self.objective(self.sample_space)

                if self.minimize:
                    ret.append((f*pdf).sum())
                else:
                    ret.append(-(f*pdf).sum())

            return np.array(ret)
        else:
            if self.minimize:
                return -self.objective.eval(self.objective.optimum) + \
                    np.array(self.y)
                # self.objective.eval(self.x)

            else:
                return self.objective.eval(self.objective.optimum) - \
                    np.array(self.y)
                # self.objective.eval(self.x)
