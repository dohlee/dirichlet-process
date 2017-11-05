import argparse
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

HP_MEAN = 0
HP_VAR = 0.1
CLUSTER_VAR = 0.01
ALPHA = 0.01

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help='Input data.')

	return parser.parse_args()

def sample_theta_from_conditional(data, thetas, i):
	y, theta = data[i], thetas[i]
	qs = np.array([norm(theta, CLUSTER_VAR).pdf(y) for j, theta in enumerate(thetas) if i != j])

	posteriorSigma = 1 / (1 / HP_VAR + 1 / CLUSTER_VAR)
	posteriorMu = posteriorSigma * (HP_MEAN / HP_VAR + y / 0.01)
	pYGivenTheta = norm(theta, np.sqrt(CLUSTER_VAR)).pdf(y)
	pTheta = norm(HP_MEAN, np.sqrt(HP_VAR)).pdf(theta)
	pThetaGivenY = norm(posteriorMu, np.sqrt(posteriorSigma)).pdf(theta)
	r = ALPHA * pYGivenTheta * pTheta / pThetaGivenY

	normalization = np.sum(qs) + r
	qs = qs / normalization
	r = r / normalization

	return np.random.choice(np.hstack([thetas[:i], thetas[i+1:], [norm(posteriorMu, np.sqrt(posteriorSigma)).rvs()]]), p=np.hstack([qs, [r]]))

data = np.loadtxt('../../test-data/data/1d-cluster-3.tsv')
thetas = np.random.normal(loc=0, scale=0.1, size=len(data))

for _ in range(10):
	print('Iteration %d' % (_+1))
	for i in range(len(data)):
		thetas[i] = sample_theta_from_conditional(data, thetas, i)

plt.hist(thetas, bins=20)
plt.show()