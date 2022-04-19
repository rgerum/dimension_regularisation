import numpy as np
from tensorflow import keras
import tensorflow as tf
print(tf.version.VERSION)

from dimension_regularisation.dim_includes import getOutputPath
from dimension_regularisation.dimension_reg_layer import DimensionReg, get_alpha_from_lambdas, get_alpha
from dimension_regularisation.dimensions_reg_layer_gamma import DimensionRegGammaWeights, DimensionRegGammaWeightsPreComputedBase, CalcEigenVectors
from dimension_regularisation.callbacks import SaveHistory, SlurmJobSubmitterStatus
from dimension_regularisation.robustness import get_robustness_metrics
from dimension_regularisation.attack_tf import get_attack_metrics
from dimension_regularisation.dim_includes import command_line_parameters as p
from dimension_regularisation.pca_variance import get_eigen_vectors, get_eigen_values_from_vectors, get_pca_variance
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, "mnist").load_data()

x_train = x_train/255.
data = x_train.reshape(-1, 28*28)

@tf.function
def get_pca_varianceX(data):
    """ calculate the eigenvalues of the covariance matrix """
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    # Finding the Eigen Values and Vectors for the data
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)
    eigen_values, eigen_vectors = tf.linalg.eigh(sigma)

    # resort (from big to small) and normalize sum to 1
    return eigen_values[::-1] / tf.reduce_sum(eigen_values)

@tf.function
def get_eigen_vectorsX(data):
    """ calculate the eigenvalues of the covariance matrix """
    normalized_data = data - tf.reduce_mean(data, axis=0)[None]
    # Finding the Eigen Values and Vectors for the data
    sigma = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1)
    eigen_values, eigen_vectors = tf.linalg.eigh(sigma)

    # return the eigenvectors
    return eigen_vectors


pca = PCA()
pca.fit(data)


#print(pca.explained_variance_ratio_[:10])
#print(get_pca_variance(data)[:10])

eigen_vectors = get_eigen_vectors(data)

plt.subplot(121)
N = 15
cmap = plt.get_cmap("viridis", N)
alphas = []
alphas_std = []
positions = []
for i in range(1, N):
    a = []
    for ii in range(100):
        indices = np.random.randint(0, data.shape[0], size=i)
        values = get_eigen_values_from_vectors(data[indices], eigen_vectors)
        a.append(get_alpha_from_lambdas(values))
    plt.loglog(values, color=cmap(i))
    alphas.append(np.mean(a))
    alphas_std.append(np.std(a))
    positions.append(str(i))

alphas.append(get_alpha(data))
alphas_std.append(0)
positions.append("∞")

alphas = np.array(alphas)
alphas_std = np.array(alphas_std)

plt.loglog(pca.explained_variance_ratio_, '--k')
plt.loglog(get_pca_variance(data), ':r')

plt.subplot(122)
plt.plot(positions[:-1], alphas[:-1])
plt.fill_between(positions[:-1], alphas[:-1]-alphas_std[:-1], alphas[:-1]+alphas_std[:-1], color="C0", alpha=0.5)
plt.axhline(alphas[-1], color="k")
plt.show()

