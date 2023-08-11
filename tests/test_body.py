from unittest import TestCase

from nonGenCom.Variables.Body import Body


class TestBody(TestCase):

    def test_posterior(self):
        biolsex = Body("tests/resources/body_fc_sceneries.csv", )

        expected = load_fc_mp_indexed_file("tests/resources/biolsex_posterior_examples.csv")["Posterior1"]
        obtained = biolsex.get_fc_score()

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for fc_value, mp_value in expected.index:
            self.assertAlmostEqual(expected[fc_value][mp_value], obtained[fc_value][mp_value], msg="different results")