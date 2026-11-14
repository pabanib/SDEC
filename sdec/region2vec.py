# -*- coding: utf-8 -*-
"""
Region2Vec (Liang et al., 2022, 2025), adapted to areas described by panels of series without flow data.

Code written in Keras/TensorFlow following the method and its public reference code (github.com/GeoDS/Region2vec,
code/models.py, layers.py, train.py, utils.py and analytics.py), with its defaults: two graph convolutional layers
(16 ReLU units, dropout 0.1, then 14 linear units) on the row-normalized adjacency with self-loops; loss 'divreg'
computed on the pairwise Euclidean distances of the raw output,
    L = (N_neg / N_pos) * sum_{pos} d_ij f_ij / ( sum_{neg} d_ij + sum_{ij} d_ij h_ij ),
where the positive pairs are those with a spatial interaction (f_ij > 0, f the log-transformed and min-max normalized
flow), the negative pairs those without it, and h_ij = 1 / (log(hop_ij) + 1) for pairs at 5 or more hops (0 otherwise);
Adam with learning rate 0.001 and weight decay 5e-4, up to 500 epochs, early stopping after epoch 250 when the loss
exceeds the mean of the last 50 losses, and the embedding of the epoch with the lowest loss from epoch 200 onward;
the regions are obtained with Ward's method constrained by the adjacency matrix applied to that embedding.

Adaptation: there are no flows between departments, so the spatial interaction is the neighbor relation of the
k-nearest-neighbor matrix W used by every method (f_ij = 1 for neighbors, 0 otherwise), the hop counts are those of
that graph and the node features are the standardized series (n x 2T), as for SDCN and the graph autoencoder.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
import tensorflow as tf
import keras
from sklearn.cluster import AgglomerativeClustering


def region2vec(X, W, K, semilla, ocultas=16, salida=14, dropout=0.1, lr=1e-3, decaimiento=5e-4, epocas=500,
               paciencia=50, desde=200, saltos=5):
    np.random.seed(semilla)
    keras.utils.set_random_seed(semilla)
    n, d = X.shape
    A = sp.csr_matrix(W)
    A = ((A + A.T) > 0).astype('float32')
    A.setdiag(0)
    A.eliminate_zeros()
    A1 = A + sp.eye(n, dtype='float32')
    adj = tf.constant((sp.diags(1.0 / np.asarray(A1.sum(1)).ravel()) @ A1).toarray(), tf.float32)
    pos = A.toarray()                                       # interaction: neighbors (flow 1, normalized 1)
    neg = 1.0 - pos
    np.fill_diagonal(neg, 0.0)
    hop = shortest_path(A, unweighted=True, directed=False)
    hop[~np.isfinite(hop)] = hop[np.isfinite(hop)].max()
    hops_m = np.where(hop >= saltos, 1.0 / (np.log(np.maximum(hop, 1.0)) + 1.0), 0.0)
    pos_t, neg_t, hop_t = (tf.constant(m, tf.float32) for m in (pos, neg, hops_m))
    reg = float(neg.sum() / pos.sum())
    x = tf.constant(X, tf.float32)

    def unif(fan_out):
        b = 1.0 / np.sqrt(fan_out)                         # pygcn GraphConvolution.reset_parameters
        return keras.initializers.RandomUniform(-b, b)
    W1 = tf.Variable(unif(ocultas)((d, ocultas)))
    b1 = tf.Variable(unif(ocultas)((ocultas,)))
    W2 = tf.Variable(unif(salida)((ocultas, salida)))
    b2 = tf.Variable(unif(salida)((salida,)))
    pesos = [W1, b1, W2, b2]
    opt = keras.optimizers.Adam(learning_rate=lr)

    def adelante(entrenando):
        h = tf.nn.relu(adj @ (x @ W1) + b1)
        if entrenando:
            h = tf.nn.dropout(h, rate=dropout)
        return adj @ (h @ W2) + b2

    def perdida(z):
        dif = z[:, None, :] - z[None, :, :]
        pdist = tf.sqrt(tf.reduce_sum(dif * dif, 2) + 1e-12)
        num = tf.reduce_sum(pdist * pos_t) * reg
        den = tf.reduce_sum(pdist * neg_t) + tf.reduce_sum(pdist * hop_t)
        return num / (den + 1e-15)

    hist, mejor, mejor_z = [], np.inf, None
    for ep in range(epocas):
        with tf.GradientTape() as cinta:
            z = adelante(True)
            l = perdida(z) + decaimiento * 0.5 * tf.add_n([tf.reduce_sum(p * p) for p in pesos])  # Adam weight decay (L2)
        opt.apply_gradients(zip(cinta.gradient(l, pesos), pesos))
        lv = float(l)
        if ep >= desde and lv < mejor:
            mejor, mejor_z = lv, adelante(False).numpy()
        if ep > 250 and lv > np.mean(hist[-paciencia:]):
            break
        hist.append(lv)
    if mejor_z is None:
        mejor_z = adelante(False).numpy()
    g = AgglomerativeClustering(K, connectivity=A, linkage='ward').fit_predict(mejor_z)
    return {'region2vec': g, 'region2vec_epocas': ep + 1}
