import attr
import numpy as np
import GPy
import logging

logger = logging.getLogger(__name__)

@attr.s
class Modeler():

    objective = attr.ib()
    noiseless = attr.ib(default=False)
    resample = attr.ib(default=None)
    kernel = attr.ib(default=None)

    def input_space(self, N=50, rng=None):
        """generate values at which to evaluate the objective"""

        return self.objective.sample_space(N, rng)
        
    def transform(self, x, reverse=False):
        z = x.copy()

        for i, (l, u) in enumerate(self.objective.range):
            if reverse:
                z[:, i] = z[:, i] * (u - l) + l
            else:
                z[:, i] = (z[:, i] - l) / (u - l)

        return z

    def build(self, x, y, resample = None, transform=True):
        x = np.array(x)
        y = np.array(y)
        if y.ndim == 1:
            y = y[:, None]

        if resample is None:
            resample = self.resample if not self.resample is None else 1

        if x.ndim == 1:
            x = x[None, :]

        if transform:
            x = self.transform(x)

        if y.shape[0] > 1:
            y = (y-y.mean())/y.std()
        else:
            y = y-y

        kernel = self.kernel
        if kernel is None:
            kernel = GPy.kern.RBF(x.shape[1], ARD=True)
        kernel = kernel.copy()
        model = GPy.models.GPRegression(x, y, kernel)

        logger.info('training model with {} resamples'.format(resample))

        for i in range(resample):
            kernel = kernel.copy()
            kernel.randomize()
               
            nmodel = GPy.models.GPRegression(x, y, kernel)

            if self.noiseless:
                nmodel.likelihood.variance = 0
                nmodel.likelihood.variance.fix()
        
            nmodel.optimize()

            if model is None or nmodel.log_likelihood() > model.log_likelihood():
                model = nmodel

        logger.debug("chose kernel {}".format(str(model.kern)))

        logger.info('finished training model')

        return model


