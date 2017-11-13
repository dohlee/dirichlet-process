import os
import argparse
import logging
import numpy as np
from distributions import NormalDistribution
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

def sample_theta_from_conditional(data, thetas, hp, i, baseMeasure):
    """Sample ith from distribution of thetas conditioned by
    all the other thetas and ith data point.
    """
    CLUSTER_VAR = hp['CLUSTER_VAR']
    ALPHA = hp['ALPHA']
    DIMENSION = hp['DIMENSION']

    # y is the data point that we focus on.
    # theta is the parameters that we are going to update.
    y, theta = data[i], thetas[i]
    clusterDistribution = NormalDistribution(mean=theta, cov=CLUSTER_VAR)
    qs = np.array([clusterDistribution.pdf(y) for j, theta in enumerate(thetas) if i != j])

    H = baseMeasure.posterior_distribution(data=[y], clusterVariance=CLUSTER_VAR)
    theta = np.zeros(DIMENSION)  # Value of theta doesn't matter.
    logLikelihood = clusterDistribution.logpdf(y)
    prior = baseMeasure.logpdf(theta)
    posterior = H.logpdf(theta)
    r = ALPHA * np.exp(logLikelihood + prior - posterior)

    normalization = np.sum(qs) + r
    qs = qs / normalization
    r = r / normalization

    choices = list(range(i)) + list(range(i+1, len(thetas))) + [len(thetas)]
    probabilities = np.hstack([qs, [r]])
    choice = np.random.choice(choices, p=probabilities)

    if choice == len(thetas):
        return H.rvs()
    else:
        return thetas[choice]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input data.')
    parser.add_argument('-n', '--numiter', type=int, default=10, help='Number of iteration.')
    parser.add_argument('-c', '--clustervar', type=float, required=True, help='(Hyperparameter) Cluster variance.')
    parser.add_argument('-a', '--alpha', default=0.1, help='(Hyperparameter) Inverse variance of dirichlet process.')
    parser.add_argument('-m', '--hpmean', default=np.array([0.0]), help='(Hyperparameter) Mean of base measure.')
    parser.add_argument('-r', '--hpvar', default=np.array([[1.0]]), help='(Hyperparameter) Variance of base measure.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Increase verbosity.')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    data = np.loadtxt(args.input, dtype=np.float32)

    # Prepare hyperparameters
    dimension = len(data[0]) if data.ndim > 1 else 1
    cov = args.clustervar * np.eye(dimension)
    hpmean = np.zeros([dimension])
    hpvar = 1 * np.eye(dimension)
    hyperparameters = {'ALPHA': args.alpha,
                        'CLUSTER_VAR': cov,
                        'HP_MEAN': hpmean,
                        'HP_VAR': hpvar,
                        'NUM_DATA': len(data),
                        'DIMENSION': dimension} 

    data = np.loadtxt(args.input)
    thetas = np.random.normal(loc=args.hpmean, scale=args.hpvar, size=[len(data), dimension])
    baseMeasure = NormalDistribution(mean=hpmean, cov=hpvar)

    for iteration in range(args.numiter):
        logging.info('Iteration %d' % (iteration + 1))
        for i in range(len(data)):
            thetas[i] = sample_theta_from_conditional(data, thetas, hyperparameters, i, baseMeasure)

        thetaCounts = Counter(list(map(lambda x: x[0], thetas)))
        plt.xlim((-4, 4))
        plt.hist(data, histtype='bar', bins=len(data) // 5, ec='black')
        for theta in thetas:
            plt.scatter(theta[0], y=0, alpha=0.66, zorder=2, s=thetaCounts[theta[0]] * 5)

        imgFileName = '../figures/alg1_numpy_%s_iteration_%d.png' % (os.path.basename(args.input).strip('.tsv'), iteration + 1)
        plt.savefig(imgFileName)
        logging.info('Figure %s saved.' % imgFileName)
        plt.clf()

if __name__ == '__main__':
    main()

