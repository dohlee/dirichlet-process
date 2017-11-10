import os
import argparse
import logging
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white');
from alg2_numpy import NormalDistribution

class Cluster:

    def __init__(self, clusterId, hyperparameters, distribution=NormalDistribution, baseMeasure=NormalDistribution):
        self.id = clusterId
        self.hp = hyperparameters

        self.distribution = distribution
        self.baseMeasure = baseMeasure(mean=self.hp['HP_MEAN'], stddev=np.sqrt(self.hp['HP_VAR']))
        self.data = OrderedDict()

    def add_datum(self, id, datum):
        self.data[id] = datum

    def remove_datum(self, id):
        del self.data[id]

    def is_empty(self):
        return len(self.data) == 0

    def score(self, datum):
        """Return cluster assignment score given datum."""
        phi = 0  # The value of phi doesn't matter.
        likelihood = self.distribution(mean=phi, stddev=np.sqrt(self.hp['CLUSTER_VAR'])).logpdf(datum)
        prior = self.baseMeasure.posterior_distribution(data=list(self.data.values()), clusterVariance=self.hp['CLUSTER_VAR']).logpdf(phi)
        posterior = self.baseMeasure.posterior_distribution(data=list(self.data.values()) + [datum], clusterVariance=self.hp['CLUSTER_VAR']).logpdf(phi)

        dominance = np.log(len(self.data) / (self.hp['NUM_DATA']))
        integral = likelihood + prior - posterior

        return dominance + integral

class State:

    def __init__(self, data, hyperparameters, baseMeasure=NormalDistribution, clusterDistribution=NormalDistribution, initNumCluster=1):
        self.data = data
        self.hp = hyperparameters
        self.alpha = self.hp['ALPHA']

        self.baseMeasure = baseMeasure(mean=self.hp['HP_MEAN'], stddev=np.sqrt(self.hp['HP_VAR']))
        self.clusterDist = clusterDistribution

        self.numCluster = initNumCluster
        self.maxClusterId = initNumCluster
        self._initialize()

    def _initialize(self):
        self.assignment = np.zeros([len(self.data)])
        self.clusters = OrderedDict([(clusterId, Cluster(clusterId, self.hp, distribution=self.clusterDist, baseMeasure=self.baseMeasure))\
                                    for clusterId in range(self.numCluster)])

        # Assign each datum randomly to the cluster.
        for datumId, datum in enumerate(self.data):
            assignedClusterId = np.random.choice(range(self.numCluster))
            self.clusters[assignedClusterId].add_datum(datumId, datum)
            self.assignment[datumId] = assignedClusterId

    def gibbs_step(self):
        for datumId, datum in enumerate(self.data):
            assignedCluster = self.get_assigned_cluster(datumId)
            assignedCluster.remove_datum(datumId)
            self.try_clean_up_cluster(self.assignment[datumId])

            ids = np.array([clusterId for clusterId in self.clusters.keys()] + [self.maxClusterId])
            scores = np.array([np.exp(cluster.score(datum)) for cluster in self.clusters.values()] + [np.exp(self.new_assignment_score(datum))])
            probabilities = scores / np.sum(scores)
            logging.debug(probabilities)

            # Select new assignment based on probabilities.
            assignedClusterId = np.random.choice(ids, p=probabilities)
            # Make new cluster.
            if assignedClusterId == self.maxClusterId:
                newCluster = Cluster(self.maxClusterId, self.hp, distribution=self.clusterDist, baseMeasure=self.baseMeasure)
                self.clusters[self.maxClusterId] = newCluster
                newCluster.add_datum(datumId, datum)

                self.assignment[datumId] = assignedClusterId
                self.maxClusterId += 1
                self.numCluster += 1
            # Assign to existing cluster.
            else:
                assignedCluster = self.clusters[assignedClusterId]
                assignedCluster.add_datum(datumId, datum)
                self.assignment[datumId] = assignedClusterId

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

    def try_clean_up_cluster(self, clusterId):
        if self.clusters[clusterId].is_empty():
            self.numCluster -= 1
            del self.clusters[clusterId]

    def get_assigned_cluster(self, datumId):
        return self.clusters[self.assignment[datumId]]

    def new_assignment_score(self, datum):
        phi = 0
        likelihood = self.clusterDist(mean=phi, stddev=np.sqrt(self.hp['CLUSTER_VAR'])).logpdf(datum)
        prior = self.baseMeasure.logpdf(phi)
        posterior = self.baseMeasure.posterior_distribution(data=[datum], clusterVariance=self.hp['CLUSTER_VAR']).logpdf(phi)

        dominance = np.log(self.alpha / (len(self.data) - 1 + self.alpha))
        integral = likelihood + prior - posterior

        return dominance + integral

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input data.')
    parser.add_argument('-n', '--numiter', type=int, default=10, help='Number of iteration.')
    parser.add_argument('-c', '--clustervar', type=float, required=True, help='(Hyperparameter) Cluster variance.')
    parser.add_argument('-a', '--alpha', default=0.1, help='(Hyperparameter) Inverse variance of dirichlet process.')
    parser.add_argument('-m', '--hpmean', default=0.0, help='(Hyperparameter) Mean of base measure.')
    parser.add_argument('-r', '--hpvar', default=1.0, help='(Hyperparameter) Variance of base measure.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Increase verbosity.')
    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Run in debug mode.')

    return parser.parse_args()

def main():
    args = parse_arguments()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    data = np.loadtxt(args.input, dtype=np.float32)
    hyperparameters = {'ALPHA': args.alpha,
                        'CLUSTER_VAR': args.clustervar,
                        'HP_MEAN': args.hpmean,
                        'HP_VAR': args.hpvar,
                        'NUM_DATA': len(data)} 

    state = State(data, hyperparameters=hyperparameters, initNumCluster=1)

    for _ in range(args.numiter):
        logging.info('Iteration %d: number of cluster %d' % ((_ + 1), state.numCluster))
        state.gibbs_step()

    state.plot_clusters(numIter=args.numiter, save=os.path.join('figures/alg3_numpy/', os.path.basename(args.input).strip('.tsv') + '.png'))

if __name__ == '__main__':
    main()
