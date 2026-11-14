# -*- coding: utf-8 -*-
"""
SDEC (Spatial Deep Embedded Clustering).

Dense autoencoder with one branch per variable (T -> 80 -> 40 -> 10, ReLU), concatenation (20),
a dense layer of 50 ReLU units, a linear latent layer of DIM_LATENTE = 6 units and a decoder
(40 -> 80 -> 2T). The autoencoder is trained 100 epochs with the cosine loss. A clustering layer
(Student's t soft assignments Q) is placed on the latent layer and trained with the KL divergence
against a target P in which each unit takes the soft assignments of its neighbors (matrix W).
The partition is obtained with Ward's method constrained by the neighbor graph, applied to Q, and is
recomputed every 20 iterations; training stops when the proportion of units that change group between
two updates is below 1e-4 or after 1,000 iterations.
"""
import numpy as np
import keras
from tensorflow.keras import backend as K
from keras.layers import Layer, InputSpec, Dense, Input, concatenate
from tensorflow.keras import regularizers
from keras.models import Model
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

DIM_LATENTE = 6      # dimension of the latent space (h)
MAX_ITER = 1000      # maximum number of iterations of the refinement
INTERVALO = 20       # the target and the partition are updated every INTERVALO iterations
TOL = 1e-4           # stopping tolerance (proportion of units that change group)


@keras.saving.register_keras_serializable(package="SDEC")
class ClusteringLayer(Layer):
    """Soft assignment of each sample to the centroids with a Student's t kernel (Xie et al., 2016).

    Arguments: n_clusters (number of centroids), alpha (degrees of freedom, 1 by default).
    Input shape (n_samples, n_features); output shape (n_samples, n_clusters).
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform',
                                        name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))   # each row adds up to one
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def target_espacial(W, q):
    """Spatially smoothed target: each unit takes the (weighted) soft assignments of its neighbors."""
    weight = W.todense() @ (q ** 2 / q.sum(0))
    return np.asarray((weight.T / weight.sum(1).reshape(len(q), )).T)


def target_xie(W, q):
    """Target of Xie et al. (2016), without the neighbor term (W is not used)."""
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def target_espacial_propio(W, q):
    """Spatially smoothed target that also includes the unit's own term of the DEC target: the own q_ij^2 / f_j enters
    as one more neighbor (W + I). Used only in the ablation (variant 'neighbors + own term')."""
    weight = (W.todense() + np.eye(W.shape[0])) @ (q ** 2 / q.sum(0))
    return np.asarray((weight.T / weight.sum(1).reshape(len(q), )).T)


TARGETS = {'espacial': target_espacial, 'xie': target_xie, 'espacial_propio': target_espacial_propio}


def kl_completa(p, q):
    """KL(P||Q) summed over regions and averaged over units, evaluated on all units (the loss of the refinement)."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum() / p.shape[0])


class sdec():
    """n_clust: number of regions K; inp_shape: [T, T], periods of each variable."""

    def __init__(self, n_clust, inp_shape=[100, 100], n_enc=0):
        self.n_enc = n_clust if n_enc == 0 else n_enc    # number of centroids of the clustering layer
        self.n_clusters = n_clust
        self.inp1_shape = inp_shape[0]
        self.inp2_shape = inp_shape[1]

    def gen_modelo(self, activacion='relu'):
        """Builds the autoencoder, the encoder and the clustering model.
        activacion='linear' replaces the ReLU of every layer by the identity (linear encoder variant)."""
        inp1 = Input(shape=[self.inp1_shape, ])
        inp2 = Input(shape=[self.inp2_shape, ])
        dense1 = Dense(80, activation=activacion)(inp1)
        dense1 = Dense(40, activation=activacion)(dense1)
        dense1 = Dense(10, activation=activacion)(dense1)
        dense2 = Dense(80, activation=activacion)(inp2)
        dense2 = Dense(40, activation=activacion)(dense2)
        dense2 = Dense(10, activation=activacion)(dense2)
        concat = concatenate([dense1, dense2])
        concat = Dense(50, activation=activacion)(concat)
        enco = Dense(DIM_LATENTE)(concat)
        dense = Dense(40, activation=activacion, kernel_regularizer=regularizers.l2(0.01))(enco)
        dense = Dense(80, activation=activacion, kernel_regularizer=regularizers.l2(0.01))(dense)
        decoder = Dense(self.inp1_shape + self.inp2_shape, activation="linear")(dense)

        autoencoder = Model(inputs=[inp1, inp2], outputs=decoder)
        encoder = Model(inputs=[inp1, inp2], outputs=enco)
        autoencoder.compile(optimizer='adam', loss='cosine_similarity')
        self.autoencoder = autoencoder
        self.encoder = encoder

        clustering_layer = ClusteringLayer(self.n_enc, name='clustering')(encoder.output)
        model = Model(inputs=encoder.input, outputs=clustering_layer)
        model.compile(optimizer='adam', loss='kld')
        self.model = model

    def train_autoencoder(self, inp, out):
        self.autoencoder.fit(inp, out, epochs=100, validation_split=0.2, verbose=False)

    def ajustar_modelo(self, X, W, objetivo='espacial', final='ward', semilla=None, pesos_iniciales=None,
                       max_iter=MAX_ITER, guardar=()):
        """Refinement of the embedding and final partition.

        X: list [X1, X2] with the inputs of the two branches. W: sparse neighbor matrix (k nearest neighbors).
        objetivo: 'espacial' (target smoothed with W, SDEC), 'xie' (target of Xie et al., 2016) or 'espacial_propio'
            (neighbors and the unit's own term, W + I; ablation only).
        max_iter: maximum number of iterations (MAX_ITER = 1,000 by default; larger values only in the convergence
            check).
        guardar: iterations at which the partition is stored in self.particiones (convergence check only).
        final: 'ward' (Ward's method constrained by W, applied to Q) or 'argmax' (argmax of Q).
        semilla: seed of the initial k-means (call keras.utils.set_random_seed(semilla) before gen_modelo()).
        pesos_iniciales: weights of the model from which the refinement starts (pre-trained autoencoder);
            if None, the current weights are used.

        Sets self.y_pred (final partition), self.historial (iteration, proportion of units that change
        group), self.traza (at each update: iteration, KL between the previous target and the current Q, KL between
        the new target and Q, proportion of units whose label changes, and adjusted Rand index with the previous
        partition, which does not depend on the numbering of the groups), self.ultima_actualizacion (iteration of the last
        update) and self.n_iter (last iteration run).
        """
        if pesos_iniciales is not None:
            self.model.set_weights(pesos_iniciales)
        km = KMeans(self.n_enc, random_state=semilla)
        km.fit(self.encoder.predict(X, verbose=0))
        self.model.get_layer(name="clustering").set_weights([km.cluster_centers_])
        y_pred_last = np.copy(km.labels_)
        tgt = TARGETS[objetivo]
        agrupador = AgglomerativeClustering(self.n_clusters, connectivity=W, linkage='ward')
        self.historial = []
        self.traza = []          # (iteration, KL(P_prev||Q), KL(P||Q), change of labels, ARI with the previous partition)
        p = None
        self.particiones = {}
        ite = 0
        for ite in range(int(max_iter)):
            if ite % INTERVALO == 0:
                q = self.model.predict(X, verbose=0)
                kl_previa = kl_completa(p, q) if p is not None else None
                p = tgt(W, q)
                y_pred = agrupador.fit_predict(q) if final == 'ward' else q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_previo = y_pred_last
                y_pred_last = np.copy(y_pred)
                self.traza.append((ite, kl_previa, kl_completa(p, q), float(delta_label),
                                   float(adjusted_rand_score(y_pred_previo, y_pred)) if ite > 0 else None))
                if ite in guardar:
                    self.particiones[ite] = np.copy(y_pred)
                if ite > 0:
                    self.historial.append((ite, float(delta_label)))
                    if delta_label < TOL:
                        break
            self.model.fit(x=[X[0], X[1]], y=p, verbose=False)
        self.n_iter = ite
        self.ultima_actualizacion = self.historial[-1][0] if self.historial else None
        self.y_pred = agrupador.fit_predict(q) if final == 'ward' else q.argmax(1)
