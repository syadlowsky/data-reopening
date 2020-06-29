from scipy.special import gammaln
from numpy import log, exp, sqrt
import numpy as np
import scipy.stats

class GrowthRateConversion(object):
    def __init__(self):    
        self.lam, self.k = self.get_params(5.0, 1.9)

    def get_params(self, mu, sigma):
      # we know that mu = lambda * Gamma(1+1/k)
      # and that sqrt(sigma^2 + mu^2) = lambda * sqrt(Gamma(1+2/k))
      # so 0.5 gammaln(1+2/k) - gammaln(1+1/k) = 0.5 log(sigma^2 + mu^2) - log(mu)
      dk = .0001
      k = 10*dk
      target = 0.5 * log(sigma**2 + mu**2) - log(mu)
      def f(k):
        return 0.5 * gammaln(1+2/k) - gammaln(1+1/k)
    
      while f(k) > target:
        k += dk
    
      lam = mu / exp(gammaln(1+1/k))
      mu_check = lam * exp(gammaln(1+1/k))
      sig_check = lam * sqrt(exp(gammaln(1+2/k)) - exp(gammaln(1+1/k))**2)
      return lam, k
    
    def filter(self):
       return np.diff(scipy.stats.weibull_min.cdf(np.arange(0, 14), c=self.k, scale=self.lam))
    
    def generation_time(self, t):
      return (self.k/self.lam) * (t/self.lam) ** (self.k-1) * exp(-(t/self.lam)**self.k)
    
    def get_r(self, r_discrete):
      return log(1+r_discrete)
    
    def get_r_discrete(self, r):
      return exp(r) - 1
    
    def get_R0(self, r, dt):
      integral = 0.0
      ts = np.arange(0, 20, dt)
      gs = self.generation_time(ts)
      integrand = exp(-r * ts) * gs
      integral = np.sum(integrand * dt)
      R0 = 1 / integral
      return R0
    
    def get_r_from_R0(self, R0, dt):
      dr = .001
      r = -0.8
      while self.get_R0(r, dt) < R0:
        r += dr
      R0_check = self.get_R0(r, dt)
      return r
    
    def growth_rate(self, R0):
      return self.get_r_discrete(self.get_r_from_R0(R0, .01))
    
    def R0_ez(self, r_discrete):
      return self.get_R0(self.get_r(r_discrete), .001)

class HospitalizationFilter(object):
    def __init__(self):
        self.filter = self._pdf()

    def _pdf(self):
        # From IQR reported here: https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(20)30528-6/fulltext
        k = 1.5
        lam = 7
        symp_to_hosp = np.diff(scipy.stats.weibull_min.cdf(np.arange(-1, 25), c=k, scale=lam))
        # From appendix here: https://www.acpjournals.org/doi/10.7326/M20-0504
        mu = 1.621
        sigma = 0.418
        scale = np.exp(mu)
        incubation = np.diff(scipy.stats.lognorm.cdf(np.arange(-1, 30), s=sigma, scale=scale))
        # assume they're independent and apply convolution
        pdf = np.convolve(incubation, symp_to_hosp, mode="full")
        return pdf

    def apply_filter(self, sequence):
        filter_len = len(self.filter)
        output = np.zeros(len(sequence))
        for i in range(filter_len):
            output[i] = sequence[0:len(self.filter)].dot(self.filter)
        for i in range(filter_len, len(output)):
            output[i] = sequence[(i-len(self.filter)):i].dot(self.filter)
        return output

class LoSFilter(object):
    def __init__(self, filename="los_trunc.csv"):
        self.filename = filename
        self.los_cdf = self._read_los_distribution(filename)
        self.filter = self._convert_los_distribution_to_filter()

    def apply_filter(self, sequence):
        filter_len = len(self.filter)
        output = np.zeros(len(sequence))
        for i in range(filter_len):
            output[i] = sequence[0:len(self.filter)].dot(self.filter)
        for i in range(filter_len, len(output)):
            output[i] = sequence[(i-len(self.filter)):i].dot(self.filter)
        return output

    def _read_los_distribution(self, filename):
        return np.genfromtxt(filename, delimiter=',', skip_header=1)

    def _convert_los_distribution_to_filter(self):
        filter = self.los_cdf[::-1, 1]
        return filter

    def get_los_pdf(self):
        if self.los_cdf[0,0] > 0:
            self.los_cdf = np.insert(self.los_cdf, [0], [[0, 1.0]], axis=0)
        return -np.diff(self.los_cdf[:, 1])
