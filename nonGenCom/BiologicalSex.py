# prueba inicial de lau
import numpy as np


def biolsex(likelihood, prior):
    # todo: add documentation
    evidence = np.matmul(likelihood, prior)
    posterior = []
    for i in range(likelihood.shape[0]):
        posterior.append([])
        for j in range(likelihood.shape[1]):
            posterior[i].append(likelihood[i][j] * prior[j] / evidence[i])
    return posterior


if __name__ == '__main__':
    a = np.array([[0.8, 0.1, 0.025, 0.05, 0.025], [0.05, 0.05, 0.8, 0.05, 0.05]]).reshape((2, 5)).transpose()
    b = np.array([[0.5], [0.5]]).reshape((2, 1))

    print(biolsex(a, b))
