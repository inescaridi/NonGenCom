from unittest import TestCase

from nonGenCom.Variables.BiologicalSex import BiologicalSex
from nonGenCom.Utils import load_fc_mp_indexed_file


class TestBiologicalSex(TestCase):

    def test_posterior(self):
        biolsex = BiologicalSex("tests/resources/biolsex_context_examples.csv", "tests/resources/biolsex_scenery_examples.csv")

        expected = load_fc_mp_indexed_file("tests/resources/biolsex_posterior_examples.csv")["Posterior1"]
        obtained = biolsex.get_fc_posterior("Context1", "Scenery1")

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for fc_value, mp_value in expected.index:
            self.assertAlmostEqual(expected[fc_value][mp_value], obtained[fc_value][mp_value], msg="different results")
