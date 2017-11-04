import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')

ALPHA = 10
STDDEV = 0.3
NUM_DATA = 100
COLORS = ['red', 'green', 'blue', 'black', 'purple']

def generate_test_data(numData, numCluster):
	"""Generate test data with numData points forming
	numCluster clusters.
	"""
	# Mixing coefficient.
	pi = np.random.dirichlet([ALPHA] * numCluster)
	# Means of clusters.
	mus = np.linspace(start=0, stop=numCluster, num=numCluster) + np.random.normal(loc=0, scale=0.1, size=numCluster)
	# Generate and return data.
	data, labels = [], []
	for _ in range(numData):
		mu = np.random.choice(mus, p=pi)
		d = np.random.normal(loc=mu, scale=STDDEV)
		data.append(d)
		labels.append(np.where(mus == mu)[0])

	data, labels = np.array(data), np.array(labels)
	return data, labels.flatten()

if __name__ == '__main__':
	for numCluster in range(2, 6):
		data, labels = generate_test_data(numData=NUM_DATA * numCluster, numCluster=numCluster)

		# Save figures.
		plt.clf()
		for i, label in enumerate(range(numCluster)):
			plt.hist(data[labels == label], color=COLORS[i], bins=20, alpha=0.66)
		plt.savefig('../figures/1d-cluster-%d.png' % numCluster)

		# Save data as tsv file.
		df = pd.DataFrame(data)
		df.to_csv('../data/1d-cluster-%d.tsv' % numCluster, sep='\t', index=False, header=False)