import numpy as np
from pandas import DataFrame


def biolsex(context: DataFrame, scenery: DataFrame) -> DataFrame:
    """
    Computes the score for biological sex, given:
    :param likelihood: an np.array of shape (n, m) representing P(FC= x | MP= y) TODO new likelihood['M']['F'] FC=M MP=F
    :param prior: an np.array of shape (m, 1) representing P(MP= y) TODO new prior['F'] MP=F
    :return: posterior: an np.array of shape (n, m) representing P(FC=x |MP=y) * P(MP= x) / P(FC= y)
    TODO new return format (double index MP, FC)
    FC / MP     M   F   O
    F          0.1 0.2
    PF         0.6
    I

    """
    likelihood = scenery.sort_index().unstack().values[:, :].astype(float)
    prior = context.sort_index().array.astype(float)

    l_n, l_m = likelihood.shape
    evidence = np.matmul(likelihood, prior).reshape(l_n, 1)

    prior_as_matrix = np.array(list(prior) * l_n).reshape(likelihood.shape)  # todo: low priority, find a way to avoid using list()
    posterior = likelihood * prior_as_matrix / evidence

    return posterior

