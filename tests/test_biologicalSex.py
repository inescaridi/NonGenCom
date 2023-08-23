from unittest import TestCase

from nonGenCom.Variables.BiologicalSex import BiologicalSex
from nonGenCom.Utils import load_double_indexed_indexed_file, FC_INDEX_NAME, R_INDEX_NAME


class TestBiologicalSex(TestCase):

    def test_fc_score(self):
        biolsex = BiologicalSex("Uniform", "High", "Perfect representation")

        expected = load_double_indexed_indexed_file("tests/resources/biolsex_posterior_examples.csv", 'FC',
                                                    FC_INDEX_NAME, 'R', R_INDEX_NAME)["Posterior1"]
        obtained = biolsex.get_fc_score()

        for fc_value, r_value in expected.index:
            self.assertAlmostEqual(expected[fc_value][r_value], obtained[fc_value][r_value], msg="different results")
