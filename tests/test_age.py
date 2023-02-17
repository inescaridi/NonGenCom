from unittest import TestCase

import pandas as pd

from nonGenCom.Age import Age


class TestAge(TestCase):

    def test_likelihood(self):
        age_var = Age()  # TODO add context examples for age

        expected = pd.read_csv("tests/resources/age_posterior_v1.csv").set_index(['FC', 'MP'])['likelihood']
        obtained = age_var.get_posterior("Standard")

        self.assertEqual(expected.shape, obtained.shape, "different shape")

        for fc_value, mp_value in expected.index:
            self.assertAlmostEqual(expected[fc_value][mp_value], obtained[fc_value][mp_value], msg="different results")
