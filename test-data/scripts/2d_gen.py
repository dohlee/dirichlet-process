import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

ALPHA = 10
COVARIANCE = np.array([[0.05, 0], [0, 0.1]])
NUM_DATA = 300
COLORS = ['red', 'green', 'blue', 'black', 'purple']

def generate_test_data(numData, numCluster):
	"""Generate test data with numData points forming
	numCluster clusters.
	"""
	# Mixing coefficient.
	pi = np.random.dirichlet([ALPHA] * numCluster)
	# Means of clusters.
	muX = np.linspace(start=0, stop=1, num=numCluster) + np.random.normal(loc=0, scale=0.01, size=numCluster)
	muY = np.linspace(start=0, stop=1, num=numCluster) + np.random.normal(loc=0, scale=0.01, size=numCluster)
	mus = np.array(list(zip(np.random.permutation(muX), np.random.permutation(muY))))
	# Generate and return data.
	data, labels = [], []
	for _ in range(numData):
		label = np.random.choice(range(len(mus)), p=pi)
		mu = mus[label]
		d = np.random.multivariate_normal(mean=mu, cov=COVARIANCE / numCluster)
		data.append(d)
		labels.append(label)

	data, labels = np.array(data), np.array(labels)
	return data, labels.flatten()

if __name__ == '__main__':
	for numCluster in range(2, 6):
		data, labels = generate_test_data(numData=NUM_DATA * numCluster, numCluster=numCluster)

		# Save figures.
		plt.clf()
		for i, label in enumerate(range(numCluster)):
			plt.scatter(data[labels == label, 0], data[labels == label, 1], color=COLORS[i], alpha=0.66)
		plt.savefig('../figures/2d-cluster-%d.png' % numCluster)

		# Save data as tsv file.
		df = pd.DataFrame(data)
		df.to_csv('../data/2d-cluster-%d.tsv' % numCluster, sep='\t', index=False, header=False)