from unittest import TestCase

import pandas as pd

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME, load_fc_indexed_file
from nonGenCom.Variables.AgeByCategory import AgeByCategory
from nonGenCom.Variables.AgeContinuous import AgeContinuous
from nonGenCom.Variables.AgeMPRange import AgeMPRange


class TestAge(TestCase):

    def test_likelihood_v1(self):
        age_var = AgeByCategory()

        expected = pd.read_csv("tests/resources/age/age_likelihood_v1.csv", dtype=str)\
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME})\
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['likelihood']
        obtained = age_var.get_FC_likelihood()

        self._compare_fc_mp_indexed(expected, obtained)

    def test_posterior_v1(self):
        age_v1 = AgeByCategory()

        expected = pd.read_csv("tests/resources/age/age_posterior_v1.csv", dtype=str)\
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME})\
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['posterior']
        obtained = age_v1.get_posterior("Standard")

        self._compare_fc_mp_indexed(expected, obtained)

    def test_posterior_v2(self):
        ranges_df = load_fc_indexed_file("nonGenCom/default_inputs/age_ranges.csv")
        default_category_ranges = ranges_df.groupby(FC_INDEX_NAME).agg({'age': (min, max)})['age'].astype(int)\
            .apply(tuple, axis=1).to_dict()

        expected = pd.read_csv("tests/resources/age/age_posterior_v1.csv", dtype=str) \
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME}) \
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['posterior'].astype(float)

        age_v2 = AgeContinuous(context_name='Standard')

        for category, age_range in default_category_ranges.items():
            age_min, age_max = age_range

            for mp_age in range(age_v2.min_age, age_v2.max_age+1):
                posterior = age_v2.get_posterior_for_case(age_min, age_max, mp_age)
                self.assertAlmostEqual(expected[category][str(mp_age)], posterior,
                                       msg=f"different results for {(category, mp_age)}")

    def test_evidence_v2(self):
        age_v2 = AgeContinuous(context_name='Standard')

        expected = pd.read_csv("tests/resources/age/age_evidence_v2.csv").set_index('FC')['Evidence']
        obtained = age_v2.evidence

        for fc_value in expected.index:
            self.assertAlmostEqual(expected.loc[fc_value], obtained.loc[fc_value],
                                   places=8,
                                   msg=f"different results for {fc_value}")

    def _compare_fc_mp_indexed(self, expected, obtained):
        expected_index_set = set(expected.index)
        obtained_index_set = set(expected.index)
        self.assertEqual(expected_index_set, obtained_index_set,
                         f"different index! missing {expected_index_set.symmetric_difference(obtained_index_set)}")

        for fc_value, mp_value in expected.index:
            expected_value = float(expected[fc_value][mp_value].replace(',', '.').replace('E', 'e'))
            obtained_value = float(obtained[fc_value][mp_value])
            self.assertAlmostEqual(expected_value, obtained_value,
                                   places=8,
                                   msg=f"different results for {(fc_value, mp_value)}")

    def test_MP_likelihood_v3(self):
        age_v3 = AgeMPRange()

        for epsilon in [0, 1, 2, 5, 10]:
            expected = pd.read_csv(f"tests/resources/age/Age_MP_likelihood_epsilon{epsilon}.csv", dtype=float, index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())
            obtained = age_v3.get_MP_likelihood(epsilon=epsilon, min_age=min_age, max_age=max_age)

            obtained = obtained.unstack()

            for mp_age in range(min_age, max_age+1):
                for r_age in range(min_age, max_age+1):
                    self.assertAlmostEqual(expected.iloc[mp_age, r_age], obtained.iloc[mp_age, r_age], places=8,
                                           msg=f"different results for {(mp_age, r_age)} with epsilon: {epsilon}")

    def test_evidences_v3(self):
        for epsilon in [0, 1]:
            age_v3 = AgeMPRange(epsilon=epsilon)

            expected = pd.read_csv(f"tests/resources/age/Age_MP_evidence_epsilon{epsilon}.csv", dtype=float, index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())

            obtained = age_v3.mp_evidence
            obtained = obtained.unstack()

            for mp_age in range(min_age, max_age + 1):
                for r_age in range(min_age, max_age + 1):
                    self.assertAlmostEqual(expected.iloc[mp_age, r_age], obtained.iloc[mp_age, r_age], places=8,
                                           msg=f"different results for {(mp_age, r_age)} with epsilon: {epsilon}")
