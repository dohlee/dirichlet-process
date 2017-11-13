import os
import argparse
import logging
import numpy as np
from collections import OrderedDict
from distributions import NormalDistribution
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')


class Cluster:
    """Parameters associated to cluster."""
    def __init__(self, id, parameters, hyperparameters, distribution=NormalDistribution):
        self.id = id
        self.numDatum = 0
        self.hp = hyperparameters
        self.alpha = self.hp['ALPHA']
        self.parameters = parameters
        self.distType = distribution
        self.distribution = distribution(**parameters)
        self.data = OrderedDict()

    def add_datum(self, datumId, datum):
        self.data[datumId] = datum
        self.numDatum += 1

    def remove_datum(self, datumId):
        del self.data[datumId]
        self.numDatum -= 1

    def is_empty(self):
        return self.numDatum == 0

    def log_score(self, datum):
        """Return cluster assignment score in log scale given datum."""
        dominance = np.log(self.numDatum / (self.hp['NUM_DATA'] - 1 + self.alpha))
        return dominance + self.log_likelihood(datum) 

    def log_likelihood(self, datum):
        """Compute the likelihood of the cluster given datum."""
        return self.distribution.logpdf(datum)

    def update_parameters(self):
        """Sample new mean from posterior to update current parameters."""
        data = np.array(list(self.data.values()))
        posteriorSampledMean = self.distribution.posterior_distribution(data=data, clusterVariance=self.hp['CLUSTER_VAR']).rvs()

        self.parameters = dict([('mean', posteriorSampledMean), ('cov', self.hp['CLUSTER_VAR'])])
        self.distribution = self.distType(**self.parameters)

class State:
    """State object representing current status of the algorithm."""
    def __init__(self, data, hyperparameters, baseMeasure=NormalDistribution, clusterDistribution=NormalDistribution, initNumCluster=2):
        self.hp = hyperparameters
        self.alpha = self.hp['ALPHA']
        self.clusterDist = clusterDistribution
        self.baseMeasure = baseMeasure(mean=self.hp['HP_MEAN'], cov=self.hp['HP_VAR'])

        self._initialize(data, initNumCluster)

    def gibbs_step(self):
        """Single step of gibbs sampling."""
        self.update_assignment()
        for clusterId, cluster in self.clusters.items():
            cluster.update_parameters()

    def plot_clusters(self, numIter, save=None):
        d = [self.data[self.assignment == clusterId] for clusterId in self.clusters]
        plt.suptitle('%s' % os.path.basename(save))
        plt.title('#Iteration=%d, #Cluster=%d' % (numIter, len(self.clusters)))
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

    def _initialize(self, data, initNumCluster):
        self.data = data
        # Make initial clusters.
        parameters = [dict([('mean', self.hp['HP_MEAN']), ('cov', self.hp['CLUSTER_VAR'])]) for _ in range(initNumCluster)]
        self.clusters = OrderedDict((clusterId, Cluster(clusterId, parameters[clusterId], self.hp)) for clusterId in range(initNumCluster))

        # Randomly assign data points to clusters.
        self.assignment = np.zeros([len(data)])
        for i in range(len(data)):
            # Randomly select cluster.
            assignedClusterId = np.random.choice(range(initNumCluster))
            self.assignment[i] = assignedClusterId
            self.clusters[assignedClusterId].add_datum(i, data[i])

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
        # Scores that existing cluster is selected.
        clusterScores = np.exp([cluster.log_score(datum) for cluster in self.clusters.values()])

        # Score that new cluster is selected.
        newClusterScore = np.exp(self.new_assignment_score(datum))

        normalization = sum(clusterScores) + newClusterScore
        probabilities = np.hstack([clusterScores / normalization, [newClusterScore / normalization]])

        # Posterior base measure with single datum.
        H = self.baseMeasure.posterior_distribution(data=[datum], clusterVariance=self.hp['CLUSTER_VAR'])

        return np.random.choice(list(self.clusters.keys()) + [self.clusterMaxId], p=probabilities), H

    def new_assignment_score(self, datum):
        phi = np.zeros(self.hp['DIMENSION'])  # Value of phi doesn't matter.
        likelihood = self.clusterDist(mean=phi, cov=self.hp['CLUSTER_VAR']).logpdf(datum)
        prior = self.baseMeasure.logpdf(phi)
        posterior = self.baseMeasure.posterior_distribution(data=[datum], clusterVariance=self.hp['CLUSTER_VAR']).logpdf(phi)

        dominance = np.log(self.alpha / (len(self.data) - 1 + self.alpha))
        integral = likelihood + prior - posterior

        return dominance + integral

    def update_assignment(self):
        for i, datum in enumerate(self.data):
            assignedClusterId = self.assignment[i]

            # First, ignore the assignment status of current datum.
            self.clusters[assignedClusterId].remove_datum(i)
            self.try_clean_up_cluster(assignedClusterId)

            # Compute new assignment of the datum.
            newClusterId, H = self.get_new_assignment(datum)

            # If new cluster should be added, sample phi from posterior distribution H,
            # and add it.
            if newClusterId == self.clusterMaxId:
                self.clusterMaxId += 1
                self.numCluster += 1
                self.assignment[i] = newClusterId

                newParameters = {'mean': H.rvs(), 'cov': self.hp['CLUSTER_VAR']}

                newPhi = Cluster(newClusterId, parameters=newParameters, hyperparameters=self.hp)
                newPhi.add_datum(i, datum)
                self.clusters[newClusterId] = newPhi
            # Assign datum to existing cluster.
            else:
                self.assignment[i] = newClusterId
                self.clusters[newClusterId].add_datum(i, datum)

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

    state = State(data, hyperparameters=hyperparameters, initNumCluster=1)

    for _ in range(args.numiter):
        logging.info('Iteration %d: number of cluster %d' % ((_ + 1), state.numCluster))
        state.gibbs_step()

    state.plot_clusters(numIter=args.numiter, save=os.path.join('figures/alg2_numpy/', os.path.basename(args.input).strip('.tsv') + '.png'))

if __name__ == '__main__':
    main()

