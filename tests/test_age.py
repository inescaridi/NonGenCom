from unittest import TestCase

import pandas as pd

from nonGenCom.Age import Age


class TestAge(TestCase):

    def test_likelihood(self):
        age_var = Age()  # TODO add test context for age

        expected = pd.read_csv("tests/resources/age_likelihood_v1.csv", dtype=str).set_index(['FC', 'MP'])['likelihood']
        min_age, max_age, category_ranges = age_var.get_category_ranges()
        obtained = age_var.get_likelihood_v1(min_age, max_age, category_ranges)

        expected_index_set = set(expected.index)
        obtained_index_set = set(expected.index)
        self.assertEqual(expected_index_set, obtained_index_set,
                         f"different index! missing {expected_index_set.symmetric_difference(obtained_index_set)}")

        for fc_value, mp_value in expected.index:
            expected_value = float(expected[fc_value][mp_value].replace(',', '.').replace('E', 'e'))
            obtained_value = float(obtained[fc_value][mp_value])
            self.assertAlmostEqual(expected_value, obtained_value,
                                   places=3,
                                   msg=f"different results for {(fc_value, mp_value)}")
