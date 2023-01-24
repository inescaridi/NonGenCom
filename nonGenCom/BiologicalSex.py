import numpy as np


def biolsex(likelihood, prior):
    """
    Computes the score for biological sex, given:
    :param likelihood: an np.array of shape (n, m) representing P(FC= x | MP= y)
    :param prior: an np.array of shape (m, 1) representing P(MP= y)
    :return: posterior: an np.array of shape (n, m) representing P(FC=x |MP=y) * P(MP= x) / P(FC= y)
    """
    l_n, l_m = likelihood.shape
    evidence = np.matmul(likelihood, prior)

    prior_as_matrix = np.array(list(prior) * l_n).reshape(likelihood.shape)  # todo: low priority, find a way to avoid using list()
    posterior = likelihood * prior_as_matrix / evidence

    return posterior

