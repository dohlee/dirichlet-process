import os
import argparse
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

HP_MEAN = 0
HP_VAR = 0.1
CLUSTER_VAR = 0.01
ALPHA = 0.01
NUM_ITER = 20

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True, help='Input data.')
	parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Increase verbosity.')

	return parser.parse_args()

def sample_theta_from_conditional(data, thetas, i):
	y, theta = data[i], thetas[i]
	qs = np.array([norm(theta, CLUSTER_VAR).pdf(y) for j, theta in enumerate(thetas) if i != j])

	posteriorSigma = 1 / (1 / HP_VAR + 1 / CLUSTER_VAR)
	posteriorMu = posteriorSigma * (HP_MEAN / HP_VAR + y / CLUSTER_VAR)
	pYGivenTheta = norm(theta, np.sqrt(CLUSTER_VAR)).pdf(y)
	pTheta = norm(HP_MEAN, np.sqrt(HP_VAR)).pdf(theta)
	pThetaGivenY = norm(posteriorMu, np.sqrt(posteriorSigma)).pdf(theta)
	r = ALPHA * pYGivenTheta * pTheta / pThetaGivenY

	normalization = np.sum(qs) + r
	qs = qs / normalization
	r = r / normalization

	return np.random.choice(np.hstack([thetas[:i], thetas[i+1:], [norm(posteriorMu, np.sqrt(posteriorSigma)).rvs()]]), p=np.hstack([qs, [r]]))

def main():
	args = parse_arguments()
	if args.verbose:
		logging.basicConfig(level=logging.INFO)

	data = np.loadtxt('../../test-data/data/1d-cluster-3.tsv')
	thetas = np.random.normal(loc=0, scale=0.1, size=len(data))

	for iteration in range(NUM_ITER):
		logging.info('Iteration %d' % (iteration + 1))
		for i in range(len(data)):
			thetas[i] = sample_theta_from_conditional(data, thetas, i)

		plt.hist(data, histtype='bar', bins=len(data) // 5, ec='black')
		plt.scatter(thetas, y=np.random.normal(loc=0, scale=0.5, size=len(thetas)), color='black', zorder=2, s=10)

		imgFileName = '../figures/alg1_numpy_%s_iteration_%d.png' % (os.path.basename(args.input).strip('.tsv'), iteration + 1)
		plt.savefig(imgFileName)
		logging.info('Image %s saved.' % imgFileName)
		plt.clf()

if __name__ == '__main__':
	main()

