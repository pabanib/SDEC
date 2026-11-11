# -*- coding: utf-8 -*-
"""Helpers shared by the experiment scripts: seeding, training of one network and refinement from given weights."""
import json
import os
import numpy as np
import keras
from sdec import deep_cluster as dc


def fijar_semilla(s):
    """Seeds numpy and keras (Python, numpy and TensorFlow generators)."""
    np.random.seed(s)
    keras.utils.set_random_seed(s)


def entrenar(X, K, T, semilla, activacion='relu', objetivo=None):
    """Builds the model with the given seed and pre-trains the autoencoder (100 epochs, cosine loss).
    X: standardized matrix (n, 2T). objetivo: matrix the autoencoder reconstructs (X by default).
    Returns the sdec object and the weights of the pre-trained model."""
    fijar_semilla(semilla)
    sc = dc.sdec(K, [T, T])
    sc.gen_modelo(activacion=activacion)
    sc.train_autoencoder([X[:, :T], X[:, T:]], X if objetivo is None else objetivo)
    return sc, sc.model.get_weights()


def refinar(sc, X, W, T, objetivo, final, semilla, pesos):
    """Refinement from the weights `pesos`, re-seeding keras so that every variant that starts from the same
    pre-trained weights uses the same sequence of random numbers. Returns (labels, last iteration)."""
    sc.model.set_weights(pesos)
    keras.utils.set_random_seed(semilla)
    sc.ajustar_modelo([X[:, :T], X[:, T:]], W, objetivo=objetivo, final=final, semilla=semilla)
    return np.asarray(sc.y_pred), sc.n_iter


def leer_jsonl(ruta):
    if not os.path.exists(ruta):
        return []
    return [json.loads(l) for l in open(ruta)]


def agregar_jsonl(ruta, fila):
    with open(ruta, 'a') as f:
        f.write(json.dumps(fila) + '\n')
