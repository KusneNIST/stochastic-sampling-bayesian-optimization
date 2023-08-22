import scipy

class MonteCarlo():
    
    def __init__(self, m, px, T=100):
        self.m = m
        self.px = px
        self.T = T
        
        self.xsamp = self.px.rvs(size=T)
        self.pfx = scipy.stats.norm(*self.m.predict(self.xsamp[:, None]))
        
    def pdf(self, z):
        return self.pfx.pdf(z).sum(0) / self.T
    
    def mean(self, select=None):
        if select is None:
            select = slice(0, self.T)

        T = (select.stop - select.start)#/select.step

        # return self.pfx.mean()[select].sum(0) / T
        return self.pfx.mean()[select].mean()

    def var(self, select=None):
        if select is None:
            select = slice(0, self.T)

        T = (select.stop - select.start)#/select.step

        # return self.pfx.var()[select].sum(0) / T
        return self.pfx.var()[select].mean() + self.pfx.mean()[select].var()
