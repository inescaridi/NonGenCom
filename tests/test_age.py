from unittest import TestCase

import pandas as pd

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME
from nonGenCom.Variables.AgeV1 import AgeV1


class TestAge(TestCase):

    def test_likelihood_v1(self):
        age_var = AgeV1()  # TODO add test context for age

        expected = pd.read_csv("tests/resources/age_likelihood_v1.csv", dtype=str)\
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME})\
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['likelihood']
        obtained = age_var.get_likelihood()

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
