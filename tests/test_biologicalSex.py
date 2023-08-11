from unittest import TestCase

from nonGenCom.Variables.BiologicalSex import BiologicalSex
from nonGenCom.Utils import load_double_indexed_indexed_file


class TestBiologicalSex(TestCase):

    def test_posterior(self):
        biolsex = BiologicalSex("tests/resources/biolsex_context_examples.csv", "tests/resources/biolsex_scenery_examples.csv")

        expected = load_double_indexed_indexed_file("tests/resources/biolsex_posterior_examples.csv", first_index,
                                                    first_index_rename, second_index, second_index_rename)["Posterior1"]
        obtained = biolsex.get_fc_score()

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for fc_value, mp_value in expected.index:
            self.assertAlmostEqual(expected[fc_value][mp_value], obtained[fc_value][mp_value], msg="different results")
