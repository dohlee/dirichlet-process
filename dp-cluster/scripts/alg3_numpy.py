import os
import argparse
import logging
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_style('white');

from distributions import NormalDistribution
from util import img2gif

class Cluster:

    def __init__(self, clusterId, hyperparameters, distribution=NormalDistribution, baseMeasure=NormalDistribution):
        self.id = clusterId
        self.hp = hyperparameters

        self.distribution = distribution
        self.baseMeasure = baseMeasure(mean=self.hp['HP_MEAN'], cov=self.hp['HP_VAR'])
        self.data = OrderedDict()

    def add_datum(self, id, datum):
        self.data[id] = datum

    def remove_datum(self, id):
        del self.data[id]

    def is_empty(self):
        return len(self.data) == 0

    def score(self, datum):
        """Return cluster assignment score given datum."""
        phi = np.zeros(self.hp['DIMENSION'])  # Value of phi doesn't matter.
        likelihood = self.distribution(mean=phi, cov=self.hp['CLUSTER_VAR']).logpdf(datum)
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

        self.baseMeasure = baseMeasure(mean=self.hp['HP_MEAN'], cov=self.hp['HP_VAR'])
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

    def plot_clusters(self, iteration, save=None):
        plt.clf()

        d = [self.data[self.assignment == clusterId] for clusterId in self.clusters]
        plt.suptitle('Iteration=%d' % iteration)
        plt.title('#Cluster=%d' % len(self.clusters))
        if self.hp['DIMENSION'] == 1:
            plt.hist(d, histtype='stepfilled', alpha=.66, bins=60, ec='black')
        else:
            for data in d:
                plt.scatter(data[:, 0], data[:, 1], marker='.', alpha=0.66)

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
        phi = np.zeros(self.hp['DIMENSION'])  # Value of phi doesn't matter.
        likelihood = self.clusterDist(mean=phi, cov=self.hp['CLUSTER_VAR']).logpdf(datum)
        prior = self.baseMeasure.logpdf(phi)
        posterior = self.baseMeasure.posterior_distribution(data=[datum], clusterVariance=self.hp['CLUSTER_VAR']).logpdf(phi)

        dominance = np.log(self.alpha / (len(self.data) - 1 + self.alpha))
        integral = likelihood + prior - posterior

        return dominance + integral

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input data.')
    parser.add_argument('-o', '--output', default='../figures/alg3_numpy/', help='Output directory')
    parser.add_argument('-n', '--numiter', type=int, default=10, help='Number of iteration.')
    parser.add_argument('-c', '--clustervar', type=float, required=True, help='(Hyperparameter) Cluster variance.')
    parser.add_argument('-a', '--alpha', default=0.01, help='(Hyperparameter) Inverse variance of dirichlet process.')
    parser.add_argument('-m', '--hpmean', default=np.array([0.0]), help='(Hyperparameter) Mean of base measure.')
    parser.add_argument('-r', '--hpvar', default=np.array([[1.0]]), help='(Hyperparameter) Variance of base measure.')
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

    state = State(data, hyperparameters=hyperparameters, initNumCluster=1)

    for iteration in range(1, args.numiter + 1):
        logging.info('Iteration %d: number of cluster %d' % (iteration, state.numCluster))
        state.gibbs_step()

        state.plot_clusters(iteration=iteration, save=os.path.join(args.output, os.path.basename(args.input).strip('.tsv') + '_%d.png' % iteration))

    imgPaths = [os.path.join(args.output, os.path.basename(args.input).strip('.tsv') + '_%d.png' % iteration) for iteration in range(1, args.numiter + 1)]
    img2gif(imgPaths, os.path.join(args.output, os.path.basename(args.input).strip('.tsv') + '.gif'), duration=0.25)

if __name__ == '__main__':
    main()
