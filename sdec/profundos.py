# -*- coding: utf-8 -*-
"""
Deep graph-based comparison methods, applied to the same inputs as SDEC: the standardized data matrix X (n x 2T) as
node features and the k-nearest-neighbor matrix W as graph.

sdcn : Structural Deep Clustering Network (Bo et al., 2020). Code written in Keras/TensorFlow following the method
       and its reference code (github.com/bdy9527/SDCN, sdcn.py, GNN.py, utils.py and data/pretrain.py), with its
       architecture and hyperparameters: autoencoder 2T-500-500-2000-10 and symmetric decoder (ReLU), pre-trained 30 epochs with the
       mean squared error (Adam, learning rate 1e-3, batches of 256); a five-layer GCN that receives the
       representations of the autoencoder layer by layer (sigma = 0.5); centroids initialized with k-means (20
       initializations) on the pre-trained embedding; 200 full-batch epochs of Adam (learning rate 1e-3) on
       0.1 KL(P||Q) + 0.01 KL(P||Z) + reconstruction error, with the target P recomputed every epoch from Q.
       The graph is W made symmetric, with self-loops, row-normalized, as in the reference code.
gae  : graph autoencoder (Kipf and Welling, 2016). Code written following the method and its reference code
       (github.com/tkipf/gae, model gcn_ae), with its defaults: two GCN layers (32 ReLU units and 16 linear units), inner-product decoder,
       weighted cross-entropy on the adjacency matrix with self-loops, Adam with learning rate 0.01, 200 epochs,
       symmetric normalization of the graph. The node embedding is then clustered.

Neither method returns regions that are connected in the neighbor graph. Each is therefore reported with its own
output and also with the final step of SDEC (Ward's method constrained by W) applied to its output.
"""
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import keras
from sklearn.cluster import KMeans, AgglomerativeClustering


def grafo_simetrico(W):
    """Binary symmetric adjacency matrix without self-loops: i and j are linked if either is a neighbor of the other."""
    A = sp.csr_matrix(W)
    A = ((A + A.T) > 0).astype('float32').tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def _lineal(n_ent, n_sal):
    """Dense layer with the default initialization of torch.nn.Linear (uniform in +-1/sqrt(n_ent))."""
    b = 1.0 / np.sqrt(n_ent)
    ini = keras.initializers.RandomUniform(-b, b)
    capa = keras.layers.Dense(n_sal, kernel_initializer=ini, bias_initializer=ini)
    capa.build((None, n_ent))
    return capa


def _gcn(n_ent, n_sal):
    """Weight matrix of a GCN layer (Glorot uniform, no bias): output = A_norm @ (H @ weight)."""
    capa = keras.layers.Dense(n_sal, use_bias=False, kernel_initializer='glorot_uniform')
    capa.build((None, n_ent))
    return capa


def _pesos(capas):
    return [v for c in capas for v in c.trainable_weights]


def _ward(Z, W, K):
    return AgglomerativeClustering(K, connectivity=W, linkage='ward').fit_predict(Z)


def sdcn(X, W, K, semilla, capas=(500, 500, 2000), n_z=10, epocas_pre=30, epocas=200, lr=1e-3, sigma=0.5):
    """Returns a dict of labels: 'sdcn' (argmax of the GCN output Z, the result reported by Bo et al.),
    'sdcn_q' (argmax of the soft assignments Q of the autoencoder branch) and 'sdcn_ward' (Ward's method
    constrained by W applied to Z)."""
    np.random.seed(semilla)
    keras.utils.set_random_seed(semilla)
    n, d = X.shape
    x = tf.constant(X, dtype=tf.float32)
    A = grafo_simetrico(W) + sp.eye(n, dtype='float32')
    A = sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A                     # row normalization
    adj = tf.constant(A.toarray(), dtype=tf.float32)

    dims = (d,) + tuple(capas) + (n_z,)
    enc = [_lineal(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
    rdims = dims[::-1]
    dec = [_lineal(rdims[i], rdims[i + 1]) for i in range(len(rdims) - 1)]

    def ae(v):
        tra = []
        h = v
        for c in enc[:-1]:
            h = tf.nn.relu(c(h))
            tra.append(h)
        z = enc[-1](h)
        h = z
        for c in dec[:-1]:
            h = tf.nn.relu(c(h))
        return dec[-1](h), tra, z

    # pre-training of the autoencoder
    opt = keras.optimizers.Adam(learning_rate=1e-3)
    pesos_ae = _pesos(enc + dec)
    rng = np.random.default_rng(semilla)
    for _ in range(epocas_pre):
        orden = rng.permutation(n)
        for i in range(0, n, 256):
            lote = tf.gather(x, orden[i:i + 256])
            with tf.GradientTape() as cinta:
                x_bar, _, _ = ae(lote)
                perdida = tf.reduce_mean(tf.square(x_bar - lote))
            opt.apply_gradients(zip(cinta.gradient(perdida, pesos_ae), pesos_ae))

    # GCN branch and centroids
    gdims = dims + (K,)
    gnn = [_gcn(gdims[i], gdims[i + 1]) for i in range(len(gdims) - 1)]
    km = KMeans(K, n_init=20, random_state=semilla).fit(ae(x)[2].numpy())
    centros = keras.layers.Dense(n_z, use_bias=False)                        # holder of the K x n_z centroids
    centros.build((None, K))
    centros.set_weights([km.cluster_centers_.astype('float32')])

    def adelante():
        x_bar, tra, z = ae(x)
        h = tf.nn.relu(adj @ gnn[0](x))
        for c, t in zip(gnn[1:-1], tra):
            h = tf.nn.relu(adj @ c((1 - sigma) * h + sigma * t))
        h = adj @ gnn[-1]((1 - sigma) * h + sigma * z)
        mu = centros.trainable_weights[0]
        q = 1.0 / (1.0 + tf.reduce_sum(tf.square(tf.expand_dims(z, 1) - mu), 2))     # Student's t, v = 1
        q = q / tf.reduce_sum(q, 1, keepdims=True)
        return x_bar, q, h

    opt = keras.optimizers.Adam(learning_rate=lr)
    pesos = pesos_ae + _pesos(gnn) + centros.trainable_weights
    for _ in range(epocas):
        _, q0, _ = adelante()
        w = q0 ** 2 / tf.reduce_sum(q0, 0)
        p = tf.stop_gradient(w / tf.reduce_sum(w, 1, keepdims=True))
        with tf.GradientTape() as cinta:
            x_bar, q, h = adelante()
            plogp = tf.math.xlogy(p, p)
            kl = tf.reduce_sum(plogp - p * tf.math.log(q)) / n
            ce = tf.reduce_sum(plogp - p * tf.nn.log_softmax(h, axis=1)) / n
            re = tf.reduce_mean(tf.square(x_bar - x))
            perdida = 0.1 * kl + 0.01 * ce + re
        opt.apply_gradients(zip(cinta.gradient(perdida, pesos), pesos))
    _, q, h = adelante()
    pred = tf.nn.softmax(h, axis=1).numpy()
    return {'sdcn': pred.argmax(1), 'sdcn_q': q.numpy().argmax(1), 'sdcn_ward': _ward(pred, W, K)}


def gae(X, W, K, semilla, ocultas=(32, 16), epocas=200, lr=0.01):
    """Returns a dict of labels: 'gae_kmeans' (k-means on the node embedding) and 'gae_ward' (Ward's method
    constrained by W applied to the embedding)."""
    np.random.seed(semilla)
    keras.utils.set_random_seed(semilla)
    n, d = X.shape
    x = tf.constant(X, dtype=tf.float32)
    A = grafo_simetrico(W)
    A_ = A + sp.eye(n, dtype='float32')
    r = sp.diags(np.power(np.asarray(A_.sum(1)).ravel(), -0.5))
    adj = tf.constant((r @ A_ @ r).toarray(), dtype=tf.float32)               # symmetric normalization
    etiquetas = tf.constant(A_.toarray(), dtype=tf.float32)
    s = float(A.sum())
    peso_pos = (n * n - s) / s
    norma = n * n / ((n * n - s) * 2.0)

    c1, c2 = _gcn(d, ocultas[0]), _gcn(ocultas[0], ocultas[1])

    def codificar():
        return adj @ c2(tf.nn.relu(adj @ c1(x)))

    opt = keras.optimizers.Adam(learning_rate=lr)
    pesos = _pesos([c1, c2])
    for _ in range(epocas):
        with tf.GradientTape() as cinta:
            z = codificar()
            logits = z @ tf.transpose(z)
            perdida = norma * tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(labels=etiquetas, logits=logits, pos_weight=peso_pos))
        opt.apply_gradients(zip(cinta.gradient(perdida, pesos), pesos))
    z = codificar().numpy()
    return {'gae_kmeans': KMeans(K, n_init=20, random_state=semilla).fit_predict(z), 'gae_ward': _ward(z, W, K)}
