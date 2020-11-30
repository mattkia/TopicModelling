import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


theta_matrix = np.loadtxt('../results/theta.txt')
print(theta_matrix.shape)

n_clusters = 20
k_means = KMeans(n_clusters=n_clusters)

k_means.fit(theta_matrix)

labels = k_means.labels_

pca = PCA(n_components=2)
new_theta = pca.fit_transform(theta_matrix)

plt.scatter(new_theta[:, 0], new_theta[:, 1], c=labels)
plt.xlabel('Summarized Topic 1')
plt.ylabel('Summarized Topic 2')
plt.title(f'Clusters : {n_clusters} - Distribution of Documents Based on Their Topics')
plt.show()


