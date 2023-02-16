from unittest import TestCase

from nonGenCom.BiologicalSex import biolsex
from nonGenCom.Config import TestConfig


class TestBiologicalSex(TestCase):

    def test_biolsex(self):
        config = TestConfig()
        context = config.get_context("Context1")
        scenery = config.get_scenery("Scenery1")

        expected = config.get_posterior("Posterior1")

        obtained = biolsex(context, scenery)

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for fc_value, mp_value in expected.index:
            self.assertAlmostEqual(expected[fc_value][mp_value], obtained[fc_value][mp_value], msg="different results")
