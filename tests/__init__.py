import numpy as np
from nonGenCom.BiologicalSex import biolsex

def first_test():
    likelihood = np.array([[0.8,   0.05],
                  [0.1,   0.05],
                  [0.025, 0.8],
                  [0.05,  0.05],
                  [0.025, 0.05]])

    prior = np.array([[0.5], [0.5]])

    expected = np.array([[0.94117647, 0.05882353],
                         [0.66666667, 0.33333333],
                         [0.03030303, 0.96969697],
                         [0.5,        0.5],
                         [0.33333333, 0.66666667]])

    obtained = biolsex(likelihood, prior)

    assert obtained.shape == expected.shape, "No tienen el mismo shape"
    # assert np.array_equal(obtained, expected), "No tienen mismos elementos"  #todo:ver xq este falla
    assert np.allclose(obtained, expected), "No son parecidos parecidos"

if __name__ == "__main__":
    first_test()
    print("Everything passed")
