from abc import ABCMeta, abstractmethod
import numpy as np
import GPy
import scipy
import attr
import logging
from scipy import linalg
from scipy.linalg import lapack, blas
from scipy import special

logger = logging.getLogger(__name__)


@attr.s
class Aquisition(metaclass=ABCMeta):
    delta = attr.ib(default=.1)
    sample_size = attr.ib(default=1)

    def compute(self, model, builder, samplex, sampler, minimize, n, D):
        if model is None:
            logger.debug('no model provided, choosing at random')
            return np.random.rand(len(sampler.space))

        logger.info('aquisition compute with D={} in iteration {}'.format(D, n))

        mu, var = model.predict(builder.transform(samplex))
        mu = mu[:, 0]
        var = var[:, 0]

        logger.debug("running precompute...")
        self._precompute(model, builder, samplex, mu, var, minimize, n, D)

        logger.debug("calculating acquisitions...")
        aq = []
        # for hp in sampler.space.values:
        for i in range(len(sampler.space)):
            a = self._compute(model, builder, samplex, mu, var,
                              sampler.build(i), minimize, n, D)
            aq.append(a)
        aq = np.array(aq)

        # cleanup, find any negative infinity values and replace them
        # with the minimum less a small fraction of the aquisition
        # range. essentially make these never selected values but not
        # a weird thing like infinity
        select = np.isneginf(aq)
        offset = 1e-9
        aq[select] = (1+offset)*aq[~select].min() - offset*aq[~select].max()

        if aq.ndim > 1:
            assert aq.ndim == 2
            assert aq.shape[1] == 1
            aq = aq[:, 0]

        return aq

    def beta(self, n, d):
        return np.sqrt(2 * np.log(d * n**2 * np.pi**2 / 6 / self.delta))

    def alpha(self, mu, var, n, d, minimize=True):
        "standard UCB computation"
        if mu.ndim > 1:
            mu = mu.ravel()
        if var.ndim > 1:
            var = var.ravel()

        s = np.sqrt(var)
        if minimize:
            return -mu + s*self.beta(n, d)
        return mu + s*self.beta(n, d)

    @abstractmethod
    def _precompute(self, model, builder, dist, minimize, n, D):
        "any actions needed before computing distribution aquisition values"
        pass

    @abstractmethod
    def _compute(self, model, builder, dist, minimize, n, D):
        pass

    @classmethod
    def find_aquisition(cls, o):
        for obj in cls.__subclasses__():
            if o.lower() in obj.__name__.lower():
                return obj

        return None


@attr.s
class Random(Aquisition):
    """Random sampling aquisition function."""

    def _precompute(self, *args, **kwargs):
        pass

    def _compute(self, *args, **kwargs):
        return np.random.rand()


@attr.s
class BatchMatch(Aquisition):

    # sample-size, ideally matching true sample size

    xselect = attr.ib(default=None, init=False)
    indselect = attr.ib(default=None, init=False)

    def _precompute(self, model, builder, samplex, mu, var, minimize, N, D):
        predx = builder.transform(samplex)

        mu, var = model.predict(predx)
        alpha = self.alpha(mu, var, N, D, minimize)

        x = model.X.tolist()
        y = model.Y[:, 0].tolist()
        k = model.kern

        self.xselect = []
        self.indselect = []

        for i in range(self.sample_size):
            ind = np.argmax(alpha)
            self.indselect.append(ind)
            self.xselect.append(predx[ind, :])

            mx = np.array(x + self.xselect)

            # hallucinate value
            ki, _, _, _ = GPy.util.linalg.pdinv(k.K(mx))
            var = k.K(predx) - np.dot(
                k.K(predx, mx), np.dot(
                    ki,
                    k.K(mx, predx)
                )
            )
            var = np.diag(var)

            alpha = self.alpha(mu, var, N+i+1, D, minimize)

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        x = np.array(self.xselect)

        # convert back to problem space
        x = builder.transform(x, reverse=True)

        x = samplex[self.indselect, :]

        p = []
        for i in range(x.shape[1]):

            p.append(
                np.product(
                    dist[i].pdf(x[:, i])
                )
            )

        # return np.sum(np.log(p).sum())
        return np.product(p)

        # return the probability of the selected x's under this dist
        # p = [[d.pdf(xxx) for xxx, d in zip(xx, dist)] for xx in self.xselect]
        # return np.exp(np.log(p).sum())

@attr.s
class LocalPenalty(Aquisition):
    "A locally penalized aquisition base class."

    distance = attr.ib(default=None, init=False)
    L = attr.ib(default=None, init=False)
    M = attr.ib(default=None, init=False)
    gamma = attr.ib(default=None, init=False)
    z = attr.ib(default=None, init=False)
    _alpha = attr.ib(default=None, init=False)

    def _precompute(self, model, builder, samplex, mu, var, minimize, N, D):

        x = builder.transform(samplex)
        self.distance = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x))

        dmu, _ = model.predictive_gradients(builder.transform(samplex))
        dmu = dmu[:, :, 0]
        dmu = np.sqrt(np.power(dmu, 2).sum(1))
        self.L = np.max(dmu)

        self._alpha = []
        for i in range(self.sample_size):
            if minimize:
                self._alpha.append(-mu + self.beta(N + i, D) * np.sqrt(var))
            else:
                self._alpha.append(mu + self.beta(N + i, D) * np.sqrt(var))

        mu = mu[:, None]
        var = var[:, None]

        if minimize:
            self.M = np.min(model.Y)
            self.z = np.sqrt(1/var/2) * (self.L * self.distance + self.M - mu)
        else:
            self.M = np.max(model.Y)
            self.z = np.sqrt(1/var/2) * (self.L * self.distance - self.M + mu)

        # first dimension is previous sample, second dimension is query position
        self.gamma = special.erfc(-self.z)/2

def MaxMean_LP(LocalPenalty):

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        pdf = np.prod([dist[i].pdf(samplex[:, i])
                       for i in range(samplex.shape[1])], 0)

        assert pdf.ndim < 3
        if pdf.ndim == 2:
            assert pdf.shape[1] == 1
            pdf = pdf[:, 0]

        assert mu.shape == pdf.shape, (mu.shape, pdf.shape)

        if minimize:
            mu = -mu

        a = 0
        for i in range(self.sample_size):
            a += np.nansum(pdf * mu * np.power(gamma, i))
            
        # a = np.nansum(mu * pdf)

        return a

def Mean_LP(LocalPenalty):

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):
        x = np.column_stack([d.mean() for d in dist])
        x = builder.transform(x)

        mu, var = model.predict(x)

        return self.alpha(mu, var, n, D, minimize)

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        pdf = np.prod([dist[i].pdf(samplex[:, i])
                       for i in range(samplex.shape[1])], 0)

        assert pdf.ndim < 3
        if pdf.ndim == 2:
            assert pdf.shape[1] == 1
            pdf = pdf[:, 0]

        assert mu.shape == pdf.shape, (mu.shape, pdf.shape)

        if minimize:
            mu = -mu

        a = 0
        for i in range(self.sample_size):
            a += np.nansum(pdf * mu * np.power(gamma, i))
            
        # a = np.nansum(mu * pdf)

        return a

@attr.s
class SBUCB_LocalPenalty(Aquisition):

    distance = attr.ib(default=None, init=False)
    L = attr.ib(default=None, init=False)
    M = attr.ib(default=None, init=False)
    gamma = attr.ib(default=None, init=False)
    z = attr.ib(default=None, init=False)
    _alpha = attr.ib(default=None, init=False)

    def _precompute(self, model, builder, samplex, mu, var, minimize, N, D):

        x = builder.transform(samplex)
        self.distance = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(x))

        dmu, _ = model.predictive_gradients(builder.transform(samplex))
        dmu = dmu[:, :, 0]
        dmu = np.sqrt(np.power(dmu, 2).sum(1))
        self.L = np.max(dmu)

        self._alpha = []
        for i in range(self.sample_size):
            if minimize:
                self._alpha.append(-mu + self.beta(N + i, D) * np.sqrt(var))
            else:
                self._alpha.append(mu + self.beta(N + i, D) * np.sqrt(var))

        mu = mu[:, None]
        var = var[:, None]

        if minimize:
            self.M = np.min(model.Y)
            self.z = np.sqrt(1/var/2) * (self.L * self.distance + self.M - mu)
        else:
            self.M = np.max(model.Y)
            self.z = np.sqrt(1/var/2) * (self.L * self.distance - self.M + mu)

        # first dimension is previous sample, second dimension is query position
        self.gamma = special.erfc(-self.z)/2

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        # compute the pdf for each samplex position
        pdf = np.prod([dist[i].pdf(samplex[:, i])
                       for i in range(samplex.shape[1])], 0)
        # pdf = np.prod([dist[i].pdfs for i in range(samplex.shape[1])], 0)

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


@attr.s
class StochasticBUCB(Aquisition):

    T = attr.ib(default=100)

    def _precompute(self, model, builder, samplex, dist, minimize, N, D, *args, **kwargs):
        pass

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        alphas = []

        for i in range(self.T):
            xs = np.column_stack([d.sample(self.sample_size) for d in dist])
            xs = builder.transform(xs)
            mu, var = model.predict(xs)

            x = model.X
            k = model.kern

            alpha = 0
            vd = k.Kdiag(xs)
            L, jit = StochasticBUCB.jitchol(k.K(x))

            # make upper triangular
            L = L.T

            for i in range(self.sample_size):
                # build the current batch element to compute
                xb = xs[[i], :]

                # compute variance reduction from cholesky
                v, _ = GPy.util.linalg.dtrtrs(L.T, k.K(x, xb), 1)
                v = vd[i] - v.T.dot(v)

                # add ucb value
                alpha += self.alpha(mu, v, n+i+1, D, minimize)

                # update cholesky for this observation
                L = StochasticBUCB.choleskyAddElement(
                    L, k.K(x, xb)[:, 0], k.K(xb)[0, 0] + jit)

                # add to "observations"
                x = np.row_stack((x, xb))

            alphas.append(alpha)

        return np.nanmean(alphas)

    @staticmethod
    def choleskyAddElement(L, v, s):
        """add a new symetric row/column to a matrix with (upper) cholesky factor L

        For matrix A = L^T L, find the cholesky factor of the new matrix:

        B = S^T S = | A    v |
                    | v^T  s | with vector v and scalar s.

        Rather than refactor the whole matrix, this can be computed
        efficiently as 
        S = | L vv           |
            | 0 sqrt(s - vv) | with vv = L^T \ v

        See also: Osborne, 2010, Bayesian Gaussian Processes for Sequential
        Prediction, Optimisation and Quadrature (Appendix B) """

        n = L.shape[0]
        ret = np.zeros((n+1, n+1))
        ret[:n, :n] = L

        vv, _ = GPy.util.linalg.dtrtrs(L.T, v)

        ret[:n, -1] = vv

        d = s - vv.T.dot(vv)
        # d = max(d, 1e-5)
        ret[-1, -1] = np.sqrt(d)

        return ret

    @staticmethod
    def jitchol(A, maxtries=5):
        "modify jitchol from GPy to return jitter"
        A = np.ascontiguousarray(A)
        L, info = lapack.dpotrf(A, lower=1)
        if info == 0:
            return L, 0
        else:
            diagA = np.diag(A)
            if np.any(diagA <= 0.):
                raise linalg.LinAlgError(
                    "not pd: non-positive diagonal elements")
            jitter = diagA.mean() * 1e-6
            num_tries = 1
            while num_tries <= maxtries and np.isfinite(jitter):
                try:
                    L = linalg.cholesky(
                        A + np.eye(A.shape[0]) * jitter, lower=True)
                    return L, jitter
                except:
                    jitter *= 10
                finally:
                    num_tries += 1
            raise linalg.LinAlgError(
                "not positive definite, even with jitter.")
        import traceback
        try:
            raise
        except:
            logging.warning('\n'.join(['Added jitter of {:.10e}'.format(jitter),
                                       '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
        return L, 0


@attr.s
class Independent(Aquisition):
    """A null model that assumes multiple draws have the same acquisition
value. E.g. batch penalty is not applied. Should be equivalent to the
full SBUCB aglorithm when doing sequential search."""

    def _precompute(self, *args, **kwargs):
        pass

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        # compute the pdf for each samplex position
        pdf = np.prod([dist[i].pdf(samplex[:, i])
                       for i in range(samplex.shape[1])], 0)

        # make sure nothing has gone wrong in pdf calc
        assert pdf.ndim < 3
        if pdf.ndim == 2:
            assert pdf.shape[1] == 1
            pdf = pdf[:, 0]

        alpha = self.alpha(mu, var, n, D, minimize)

        return self.sample_size * (alpha*pdf).sum()

@attr.s
class Mean(Aquisition):
    """A null model that takes the mean of each sampling dist and uses
that for the aquisition value"""

    def _precompute(self, *args, **kwargs):
        pass

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):
        x = np.column_stack([d.mean() for d in dist])
        x = builder.transform(x)

        mu, var = model.predict(x)

        return self.alpha(mu, var, n, D, minimize)

@attr.s
class MaxMean(Aquisition):
    "a null model taking the expected value of the predictive mean. should be overly exploitative"

    def _precompute(self, *args, **kwargs):
        pass

    def _compute(self, model, builder, samplex, mu, var, dist, minimize, n, D):

        pdf = np.prod([dist[i].pdf(samplex[:, i])
                       for i in range(samplex.shape[1])], 0)

        assert pdf.ndim < 3
        if pdf.ndim == 2:
            assert pdf.shape[1] == 1
            pdf = pdf[:, 0]

        assert mu.shape == pdf.shape, (mu.shape, pdf.shape)

        if minimize:
            mu = -mu

        
        a = np.nansum(mu * pdf)

        # print(np.argmax(pdf), np.argmax(mu), np.argmax(mu*pdf), a)

        return a


@attr.s
class AquisitionParse():
    name = attr.ib()
    aq = attr.ib()

    def kw(self, args):
        kw = {'sample_size': args.sample_size}
        for k, v in args._get_kwargs():
            if k[:len(self.name)] == self.name:
                n = k[len(self.name) + 1:]
                kw[n] = v
                logger.debug('building {}: {} = {}'.format(self.name, n, v))

        return kw

    def path(self, args):
        return '-'.join(['{}{}'.format(k, v) for k, v in
                         self.kw(args).items()])

    def build(self, args):
        return self.aq(**self.kw(args))


def buildParser(parser):

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

    sb = parser.add_parser('sbucb', help='stochastic batch UCB')
    ap = AquisitionParse('sb', SBUCB_LocalPenalty)
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

    # max mean

    sb = parser.add_parser('maxmean', help='maximum of predictive mean')
    ap = AquisitionParse('maxmean', MaxMean)
    sb.set_defaults(aq_parse=ap, aquisition='maxmean')
    
    # max mean lp

    sb = parser.add_parser('maxmean_lp', help='maximum of predictive mean, with local penalty calculation')
    ap = AquisitionParse('maxmean_lp', MaxMean)
    sb.set_defaults(aq_parse=ap, aquisition='maxmean_lp')
    
    return parser
