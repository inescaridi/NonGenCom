from unittest import TestCase

import numpy as np

from nonGenCom.BiologicalSex import biolsex
from nonGenCom.Config import TestConfig


class TestBiologicalSex(TestCase):

    def test_biolsex(self):
        config = TestConfig()
        context = config.getContext("Context1")
        scenery = config.getScenery("Scenery1")

        # likelihood = np.array([[0.8, 0.05],
        #                        [0.1, 0.05],
        #                        [0.025, 0.8],
        #                        [0.05, 0.05],
        #                        [0.025, 0.05]])
        #
        # prior = np.array([[0.5], [0.5]])

        expected = np.array([[0.94117647, 0.05882353],
                             [0.66666667, 0.33333333],
                             [0.03030303, 0.96969697],
                             [0.5, 0.5],
                             [0.33333333, 0.66666667]])

        obtained = biolsex(context, scenery)

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                # TODO change this to not be dependable on order
                self.assertAlmostEqual(expected[i][j], obtained[i][j], msg="different results")
        # self.assert_(np.allclose(obtained, expected), "different results")
