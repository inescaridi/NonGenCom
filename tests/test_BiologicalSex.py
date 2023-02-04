from unittest import TestCase

import numpy as np

from nonGenCom.BiologicalSex import biolsex
from nonGenCom.Config import TestConfig


class TestBiologicalSex(TestCase):

    def test_biolsex(self):
        config = TestConfig()
        context = config.getContext("Context1")
        scenery = config.getScenery("Scenery1")

        # expected = np.array([[0.9411764706, 0.05882352941],
        #                      [0.8, 0.2],
        #                      [0, 1],
        #                      [0, 1],
        #                      [0, 1]])
        expected = config.getPosterior("Posterior1")

        obtained = biolsex(context, scenery)

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for i in range(expected.shape[0]):
            for j in range(expected.shape[1]):
                # TODO change this to not be dependable on order
                self.assertAlmostEqual(expected[i][j], obtained[i][j], msg="different results")
        # self.assert_(np.allclose(obtained, expected), "different results")
