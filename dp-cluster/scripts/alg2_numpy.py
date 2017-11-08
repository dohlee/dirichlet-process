import os
import argparse
import logging
import numpy as np
from collections import OrderedDict
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

class Phi:
    """Parameters associated to cluster."""
    def __init__(self, id, parameters, hyperparameters, distribution=norm):
        self.id = id
        self.numAssociatedObservation = 0
        self.hp = hyperparameters
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
        posteriorSigma = 1 / (1 / self.hp['HP_VAR'] + self.numAssociatedObservation / self.hp['CLUSTER_VAR'])
        posteriorMu = posteriorSigma * (self.hp['HP_MEAN'] / self.hp['HP_VAR'] + np.sum(data, axis=0) / self.hp['CLUSTER_VAR'])

        self.parameters = dict([('loc', norm(loc=posteriorMu, scale=np.sqrt(posteriorSigma)).rvs()), ('scale', np.sqrt(self.hp['CLUSTER_VAR']))])
        self.distribution = self.distType(**self.parameters)

class State:
    """State object representing current status of the algorithm."""
    def __init__(self, data, hyperparameters, baseMeasure=norm, clusterDist=norm, initNumCluster=2):
        self.baseMeasure = baseMeasure
        self.clusterDist = clusterDist
        self.hp = hyperparameters

        self._initialize(data, initNumCluster)

    def _initialize(self, data, initNumCluster):
        self.data = data
        # Make initial clusters.
        parameters = [dict([('loc', 0.0), ('scale', np.sqrt(self.hp['CLUSTER_VAR']))]) for clusterId in range(initNumCluster)]
        self.clusters = OrderedDict((clusterId, Phi(clusterId, parameters[clusterId], self.hp)) for clusterId in range(initNumCluster))

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
        # Score that existing cluster is selected.
        clusterScores = np.array([cluster.numAssociatedObservation / (self.numData - 1 + self.hp['ALPHA']) * cluster.likelihood(datum) \
                         for clusterId, cluster in self.clusters.items()])

        posteriorVar = 1 / (1 / self.hp['HP_VAR'] + 1 / self.hp['CLUSTER_VAR'])
        posteriorMu = posteriorVar * (self.hp['HP_MEAN'] / self.hp['HP_VAR'] + datum / self.hp['CLUSTER_VAR'])

        phi = 0  # Value of phi is not important.
        pDatumGivenPhi = self.clusterDist(phi, np.sqrt(self.hp['CLUSTER_VAR'])).pdf(datum)
        pPhi = self.baseMeasure(self.hp['HP_MEAN'], np.sqrt(self.hp['HP_VAR'])).pdf(phi)
        pPhiGivenDatum = self.baseMeasure(posteriorMu, np.sqrt(posteriorVar)).pdf(phi)

        # Score that new cluster is selected.
        newClusterScore = self.hp['ALPHA'] / (self.numData - 1 + self.hp['ALPHA']) * pDatumGivenPhi * pPhi / pPhiGivenDatum

        normalization = sum(clusterScores) + newClusterScore
        probabilities = np.hstack([clusterScores / normalization, [newClusterScore / normalization]])

        # Posterior base measure with single datum.
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

                newParameters = {'loc': H.rvs(), 'scale': np.sqrt(self.hp['CLUSTER_VAR'])}

                newPhi = Phi(newClusterId, parameters=newParameters, hyperparameters=self.hp)
                newPhi.associate_datum(i, datum)
                self.clusters[newClusterId] = newPhi
            # Assign datum to existing cluster.
            else:
                self.assignment[i] = newClusterId
                self.clusters[newClusterId].associate_datum(i, datum)

    def gibbs_step(self):
        """Single step of gibbs sampling."""
        self.update_assignment()
        for clusterId, cluster in self.clusters.items():
            cluster.update_parameters()

    def plot_clusters(self, numIter, save=None):
        d = [self.data[self.assignment == clusterId] for clusterId in self.clusters]
        plt.suptitle('%s' % os.path.basename(save))
        plt.title('#Iteration=%d, #Cluster=%d' % (numIter, len(self.clusters)))
        plt.hist(d, histtype='stepfilled', alpha=.66, bins=60, ec='black')

        if save:
            logging.info('Figure %s saved.' % save)
            plt.savefig(save)
        else:
            logging.info('Showing clustering result.')
            plt.show()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input data.')
    parser.add_argument('-n', '--numiter', type=int, default=10, help='Number of iteration.')
    parser.add_argument('-c', '--clustervar', type=float, required=True, help='(Hyperparameter) Cluster variance.')
    parser.add_argument('-a', '--alpha', default=0.1, help='(Hyperparameter) Inverse variance of dirichlet process.')
    parser.add_argument('-m', '--hpmean', default=0.0, help='(Hyperparameter) Mean of base measure.')
    parser.add_argument('-r', '--hpvar', default=1.0, help='(Hyperparameter) Variance of base measure.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Increase verbosity.')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    data = np.loadtxt(args.input, dtype=np.float32)
    hyperparameters = {'ALPHA': args.alpha,
                        'CLUSTER_VAR': args.clustervar,
                        'HP_MEAN': args.hpmean,
                        'HP_VAR': args.hpvar} 

    state = State(data, hyperparameters=hyperparameters, initNumCluster=1)

    for _ in range(args.numiter):
        logging.info('Iteration %d: number of cluster %d' % ((_ + 1), state.numCluster))
        state.gibbs_step()

    state.plot_clusters(numIter=args.numiter, save=os.path.join('../figures/alg2_numpy/', os.path.basename(args.input).strip('.tsv') + '.png'))

if __name__ == '__main__':
    main()

