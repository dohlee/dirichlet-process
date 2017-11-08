import os
import argparse
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input data.')
    parser.add_argument('-n', '--numiter', default=10, type=int, help='Number of iteration.')
    parser.add_argument('-c', '--clustervar', type=float, required=True, help='(Hyperparameter) Cluster variance.')
    parser.add_argument('-a', '--alpha', default=0.1, help='(Hyperparameter) Inverse variance of dirichlet process.')
    parser.add_argument('-m', '--hpmean', default=0.0, help='(Hyperparameter) Mean of base measure.')
    parser.add_argument('-r', '--hpvar', default=1.0, help='(Hyperparameter) Variance of base measure.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Increase verbosity.')

    return parser.parse_args()

def sample_theta_from_conditional(data, thetas, hp, i):
    CLUSTER_VAR = hp['CLUSTER_VAR']
    ALPHA = hp['ALPHA']
    HP_MEAN = hp['HP_MEAN']
    HP_VAR = hp['HP_VAR']

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

    choices = np.hstack([thetas[:i], thetas[i+1:], [norm(posteriorMu, np.sqrt(posteriorSigma)).rvs()]])
    probabilities = np.hstack([qs, [r]])

    return np.random.choice(choices, p=probabilities)

def main():
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    hyperparameters = {'CLUSTER_VAR': args.clustervar,
                        'ALPHA': args.alpha,
                        'HP_MEAN': args.hpmean,
                        'HP_VAR': args.hpvar}

    data = np.loadtxt(args.input)
    thetas = np.random.normal(loc=args.hpmean, scale=args.hpvar, size=len(data))

    for iteration in range(args.numiter):
        logging.info('Iteration %d' % (iteration + 1))
        for i in range(len(data)):
            thetas[i] = sample_theta_from_conditional(data, thetas, hyperparameters, i)

        plt.hist(data, histtype='bar', bins=len(data) // 5, ec='black')
        plt.scatter(thetas, y=np.random.normal(loc=0, scale=0.5, size=len(thetas)), color='black', zorder=2, s=10)

        imgFileName = '../figures/alg1_numpy_%s_iteration_%d.png' % (os.path.basename(args.input).strip('.tsv'), iteration + 1)
        plt.savefig(imgFileName)
        logging.info('Figure %s saved.' % imgFileName)
        plt.clf()

if __name__ == '__main__':
    main()

