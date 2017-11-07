import os
import argparse
import logging
import numpy as np
from collections import OrderedDict
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

# Hyperparameters
HP_MEAN = 0
HP_VAR = 1
CLUSTER_VAR = 1 / 3**2
ALPHA = 0.1
NUM_ITER = 10

class Phi:
    """Parameters associated to cluster."""
    def __init__(self, id, parameters, distribution=norm):
        self.id = id
        self.numAssociatedObservation = 0
        self.parameters = parameters
        self.distType = distribution
        self.distribution = distribution(**parameters)
        self.associatedData = OrderedDict()

    def associate_datum(self, datumId, datum):
        self.associatedData[datumId] = datum
        self.numAssociatedObservation += 1

    def deassociate_datum(self, datumId):
        del self.associatedData[datumId]
        self.numAssociatedObservation -= 1

    def is_empty(self):
        return self.numAssociatedObservation == 0

    def likelihood(self, datum):
        return self.distribution.pdf(datum)

    def update_parameters(self):
        data = np.array([v for k, v in self.associatedData.items()])
        posteriorSigma = 1 / (1 / HP_VAR + self.numAssociatedObservation / CLUSTER_VAR)
        posteriorMu = posteriorSigma * (HP_MEAN / HP_VAR + np.sum(data, axis=0) / CLUSTER_VAR)

        self.parameters = dict([('loc', norm(loc=posteriorMu, scale=np.sqrt(posteriorSigma)).rvs()), ('scale', np.sqrt(CLUSTER_VAR))])
        self.distribution = self.distType(**self.parameters)

class State:
    """State object representing current status of the algorithm."""
    def __init__(self, data, baseMeasure=norm, clusterDist=norm, initNumCluster=2):
        self.baseMeasure = baseMeasure
        self.clusterDist = clusterDist

        self._initialize(data, initNumCluster)

    def _initialize(self, data, initNumCluster):
        self.data = data
        # Make initial clusters.
        parameters = [dict([('loc', 0.0), ('scale', np.sqrt(CLUSTER_VAR))]) for clusterId in range(initNumCluster)]
        self.clusters = OrderedDict((clusterId, Phi(clusterId, parameters[clusterId])) for clusterId in range(initNumCluster))

        # Randomly assign data points to clusters.
        self.assignment = np.zeros([len(data)])
        for i in range(len(data)):
            # Randomly select cluster.
            assignedClusterId = np.random.choice(range(initNumCluster))
            self.assignment[i] = assignedClusterId
            self.clusters[assignedClusterId].associate_datum(i, data[i])

        # Update cluster parameters based on assigned data.
        for clusterId, cluster in self.clusters.items():
            cluster.update_parameters()

        self.clusterMaxId = initNumCluster
        self.numCluster = initNumCluster
        self.numData = len(data)

    def try_clean_up_cluster(self, clusterId):
        if self.clusters[clusterId].is_empty():
            del self.clusters[clusterId]
            self.numCluster -= 1

    def get_new_assignment(self, datum):
        # Probability that existing cluster is selected.
        clusterScores = np.array([cluster.numAssociatedObservation / (self.numData - 1 + ALPHA) * cluster.likelihood(datum) \
                         for clusterId, cluster in self.clusters.items()])

        posteriorVar = 1 / (1 / HP_VAR + 1 / CLUSTER_VAR)
        posteriorMu = posteriorVar * (HP_MEAN / HP_VAR + datum / CLUSTER_VAR)

        phi = 0  # Value of phi is not important.
        pDatumGivenPhi = self.clusterDist(phi, np.sqrt(CLUSTER_VAR)).pdf(datum)
        pPhi = self.baseMeasure(HP_MEAN, np.sqrt(HP_VAR)).pdf(phi)
        pPhiGivenDatum = self.baseMeasure(posteriorMu, np.sqrt(posteriorVar)).pdf(phi)

        newClusterScore = ALPHA / (self.numData - 1 + ALPHA) * pDatumGivenPhi * pPhi / pPhiGivenDatum

        normalization = sum(clusterScores) + newClusterScore
        probabilities = np.hstack([clusterScores / normalization, [newClusterScore / normalization]])
        H = self.baseMeasure(posteriorMu, np.sqrt(posteriorVar))

        return np.random.choice(list(self.clusters.keys()) + [self.clusterMaxId], p=probabilities), H

    def update_assignment(self):
        for i, datum in enumerate(self.data):
            assignedClusterId = self.assignment[i]

            # First, ignore current data.
            self.clusters[assignedClusterId].deassociate_datum(i)
            self.try_clean_up_cluster(assignedClusterId)

            newClusterId, H = self.get_new_assignment(datum)

            # If new cluster should be added, sample phi from posterior distribution H,
            # and add it.
            if newClusterId == self.clusterMaxId:
                self.clusterMaxId += 1
                self.numCluster += 1
                self.assignment[i] = newClusterId

                newParameters = {'loc': H.rvs(), 'scale': np.sqrt(CLUSTER_VAR)}

                newPhi = Phi(newClusterId, parameters=newParameters)
                newPhi.associate_datum(i, datum)
                self.clusters[newClusterId] = newPhi
            # Assign datum to existing cluster.
            else:
                self.assignment[i] = newClusterId
                self.clusters[newClusterId].associate_datum(i, datum)

    def gibbs_step(self):
        self.update_assignment()
        for clusterId, cluster in self.clusters.items():
            cluster.update_parameters()

    def plot_clusters(self):
        d = [self.data[self.assignment == clusterId] for clusterId in self.clusters]

        plt.hist(d, histtype='stepfilled', alpha=.66, bins=60, ec='black')
        plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input data.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Increase verbosity.')

    return parser.parse_args()

def main():
    # args = parse_arguments()
    # if args.verbose:
        # logging.basicConfig(level=logging.INFO)

    data = np.loadtxt('../../test-data/data/1d-cluster-3.tsv', dtype=np.float32)
    state = State(data, initNumCluster=1)

    for _ in range(NUM_ITER):
        print('Iteration %d: number of cluster %d' % ((_ + 1), state.numCluster))
        state.gibbs_step()

    state.plot_clusters()
if __name__ == '__main__':
    main()

