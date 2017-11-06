import os
import argparse
import logging
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

# Hyperparameters
HP_MEAN = 0
HP_VAR = 0.1
CLUSTER_VAR = 0.01
ALPHA = 0.01
NUM_ITER = 20

class Phi:
    """Parameters associated to cluster."""
    def __init__(self, id, parameters, distribution=norm):
        self.id = id
        self.numAssociatedObservation = 0
        self.parameters = parameters
        self.distribution = distribution(*parameters)

    def associate_observation(self):
        self.numAssociatedObservation += 1

    def is_empty(self):
        return self.numAssociatedObservation == 0

    def likelihood(self, datum):
        return self.distribution.pdf(datum)


class State:
    """State object representing current status of the algorithm."""
    def __init__(self, data, initNumCluster=1):
        self._initialize(data, initNumCluster)

    def _initialize(self, data, initNumCluster):
        self.data = data
        self.assignment = np.array([np.random.choice(range(initNumCluster)) for _ in data])

        # Compute initial parameters of clusters based on random assignments of data.
        parameters = [dict([('loc', np.mean(data[self.assignment == clusterId])), ('scale', CLUSTER_VAR)] for clusterId in range(initNumCluster))]
        self.clusters = dict((clusterId, Phi(clusterId, parameters[clusterId])) for clusterId in range(initNumCluster))

        self.clusterMaxId = initNumCluster
        self.numCluster = initNumCluster
        self.numData = len(data)

    def clean_up_clusters(self):
        toRemove = [id for id, cluster in self.clusters.items() if cluster.is_empty()]
        for id in toRemove:
            del self.clusters[id]
        self.numCluster -= len(toRemove)

    def update_assignment(self):
        for i, datum in enumerate(self.data):
            assignedClusterId = self.assignment[i]
            qs = np.array([cluster.numAssociatedObservation - [0, 1][clusterId == assignedClusterId] / (self.numData - 1 + ALPHA) * cluster.likelihood(datum) \
                 for clusterId, cluster in self.clusters])

            posteriorSigma = 1 / (1 / HP_VAR + 1 / CLUSTER_VAR)
            posteriorMu = posteriorSigma * (HP_MEAN / HP_VAR + datum / CLUSTER_VAR)
            pYGivenTheta = norm(0, np.sqrt(CLUSTER_VAR)).pdf(datum)
            pTheta = norm(HP_MEAN, np.sqrt(HP_VAR)).pdf(0)
            pThetaGivenY = norm(posteriorMu, np.sqrt(posteriorSigma)).pdf(0)
            r = ALPHA * pYGivenTheta * pTheta / pThetaGivenY

            r = ALPHA / (self.numData - 1 + ALPHA) * pYGivenTheta * pTheta * pThetaGivenY

            normalization = np.sum(qs) + r
            qs = qs / normalization
            r = r / normalization

            newClusterId = np.random.choice(list(self.clusters.keys()) + [self.clusterMaxId], p=np.hstack([qs, [r]]))
            if newClusterId == self.clusterMaxId:
                self.clustermaxId += 1
                newMean = norm(posteriorMu, np.sqrt(posteriorSigma)).rvs()
                newParameters = {'loc':newMean, 'scale':CLUSTER_VAR}
                newPhi = Phi(newClusterId, parameters=newParameters)
                self.clusters[newClusterId] = newPhi
            else:
                self.assignment[i] = newClusterId
                self.clusters[newClusterId].associate_observation()

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

        plt.hist(data, bins=len(data) // 5)
        plt.scatter(thetas, y=np.random.normal(loc=0, scale=0.5, size=len(thetas)), color='black', zorder=2, s=10)

        imgFileName = '../figures/alg1_numpy_%s_iteration_%d.png' % (os.path.basename(args.input).strip('.tsv'), iteration + 1)
        plt.savefig(imgFileName)
        logging.info('Image %s saved.' % imgFileName)
        plt.clf()

if __name__ == '__main__':
    main()

