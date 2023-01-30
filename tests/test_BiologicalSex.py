from unittest import TestCase

import numpy as np

from nonGenCom.BiologicalSex import biolsex


class TestBiologicalSex(TestCase):

    def test_biolsex(self):
        likelihood = np.array([[0.8, 0.05],
                               [0.1, 0.05],
                               [0.025, 0.8],
                               [0.05, 0.05],
                               [0.025, 0.05]])

        prior = np.array([[0.2], [0.8]])

        expected = np.array([[0.94117647, 0.05882353],
                             [0.66666667, 0.33333333],
                             [0.03030303, 0.96969697],
                             [0.5, 0.5],
                             [0.33333333, 0.66666667]])

        obtained = biolsex(likelihood, prior)

        self.assertEqual(expected.shape, obtained.shape, "different shape")
        self.assert_(np.allclose(obtained, expected), "different results")
