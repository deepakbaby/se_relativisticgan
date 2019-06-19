"""
Functions used by Wasserstein GAN
"""
from keras import backend as K
import numpy as np

def wasserstein_loss(y_true, y_pred):
    """
    Define the Wasserstein loss for compiling and training the model
    """
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """
    This term is used for stabilizing the WGAN training.
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    axes_for_sum = tuple(np.arange(1, len(gradients_sqr.shape)))
    gradients_sqr_sum = K.sum(gradients_sqr, axis=axes_for_sum)
    gradient_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_norm)
    return K.mean(gradient_penalty)

