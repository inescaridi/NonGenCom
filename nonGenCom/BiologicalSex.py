# prueba inicial de lau
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


if __name__ == '__main__':
    a = np.array([[0.8, 0.1, 0.025, 0.05, 0.025], [0.05, 0.05, 0.8, 0.05, 0.05]]).reshape((2, 5)).transpose()
    b = np.array([[0.5], [0.5]]).reshape((2, 1))

    expected = np.array([[0.94117647, 0.66666667, 0.03030303, 0.50000000, 0.33333333],
                         [0.05882353, 0.33333333, 0.96969697, 0.50000000, 0.66666667]]).reshape((2, 5)).transpose()

    actual = biolsex(a, b)
    print("expected")
    print(expected)
    print("actual")
    print(actual)
