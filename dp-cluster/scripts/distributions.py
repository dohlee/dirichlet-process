import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import inv

class NormalDistribution:

    def __init__(self, mean, cov):
        self.mean = mean
        self.covariance = cov
        self.distribution = multivariate_normal(mean=mean, cov=cov)

    def rvs(self):
        """Return random sample of the distribution."""
        return self.distribution.rvs()

    def logpdf(self, x):
        """Return log probability density function."""
        return self.distribution.logpdf(x)

    def pdf(self, x):
        """Return probability density function."""
        return self.distribution.pdf(x)

    def posterior_distribution(self, data, clusterVariance):
        """Return posterior distribution of the distribution given data."""
        n = len(data)
        posteriorCovariance = inv((inv(self.covariance) + n * inv(clusterVariance)))
        posteriorMean = np.dot(posteriorCovariance, (np.dot(inv(self.covariance), self.mean) + np.dot(inv(clusterVariance), np.sum(data, axis=0))))

        return NormalDistribution(mean=posteriorMean, cov=posteriorCovariance)

    def __call__(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        return self