from unittest import TestCase

import pandas as pd

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME, load_fc_indexed_file, change_index_level_type
from nonGenCom.Variables.Age import Age


class TestAge(TestCase):
    def test_MP_likelihood(self):
        age = Age()

        for epsilon in [2]:
            expected = pd.read_csv(f"tests/resources/age/Age_MP_likelihood_epsilon{epsilon}.csv", index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())
            obtained = age.get_mp_likelihood()

            obtained = obtained.unstack()

            for mp_age in range(min_age, max_age+1):
                for r_age in range(min_age, max_age+1):
                    self.assertAlmostEqual(expected.iloc[mp_age, r_age], obtained.iloc[mp_age, r_age], places=8,
                                           msg=f"different results for {(mp_age, r_age)} with epsilon: {epsilon}")

    def test_MP_evidence(self):
        for epsilon in [0, 1]:

            expected = pd.read_csv(f"tests/resources/age/Age_MP_evidence_epsilon{epsilon}.csv", index_col=0)['Evi']
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())

            age = Age(min_age=min_age, max_age=max_age, epsilon=epsilon)
            obtained = age.mp_evidence

            for age in range(min_age, max_age + 1):
                self.assertAlmostEqual(expected.loc[age], obtained.loc[age], places=8,
                                       msg=f"different results for {age} with epsilon: {epsilon}")

    def test_FC_evidence(self):
        expected = pd.read_csv("tests/resources/age/Age_FC_evidence.csv", index_col=0)['Evi']
        min_age = int(expected.index.min())
        max_age = int(expected.index.max())
        age = Age(min_age=min_age, max_age=max_age)

        obtained = age.fc_evidence

        for age in range(min_age, max_age + 1):
            self.assertAlmostEqual(expected.loc[age], obtained.loc[age], places=8,
                                   msg=f"different results for {age}")

    def test_score_numerator(self):
        expected = pd.read_csv("tests/resources/age/Age_score_numeratos_MPepsilon2.csv", index_col=0)
        min_age = int(expected.index.min())
        max_age = int(expected.index.max())
        age = Age(min_age=min_age, max_age=max_age)

        obtained = age.score_numerator

        for fc_age in range(min_age, max_age + 1):
            for mp_age in range(min_age, max_age + 1):
                self.assertAlmostEqual(expected.iloc[fc_age, mp_age], obtained.iloc[fc_age, mp_age], places=8,
                                       msg=f"different results for {(fc_age, mp_age)}")

    def test_MP_posterior(self):
        for epsilon in [0, 1]:
            expected = pd.read_csv(f"tests/resources/age/Age_MP_posterior_epsilon{epsilon}.csv", index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())
            age = Age(min_age=min_age, max_age=max_age, epsilon=epsilon)

            for fc_age in range(min_age, max_age+1):
                for mp_age in range(min_age, max_age+1):
                    obtained = age.get_mp_score_for_range(fc_age, fc_age, mp_age, mp_age)

                    self.assertAlmostEqual(expected.iloc[fc_age, mp_age], obtained, places=8,
                                           msg=f"different results for {(fc_age, mp_age)} with epsilon: {epsilon}")

    def test_FC_posterior(self):
        for epsilon in [0, 1]:
            expected = pd.read_csv(f"tests/resources/age/Age_FC_posterior_epsilon{epsilon}.csv", index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())
            age = Age(min_age=min_age, max_age=max_age, epsilon=epsilon)

            for fc_age in range(min_age, max_age + 1):
                for mp_age in range(min_age, max_age + 1):
                    obtained = age.get_fc_score_for_range(fc_age, fc_age, mp_age, mp_age)

                    self.assertAlmostEqual(expected.iloc[fc_age, mp_age], obtained, places=8,
                                           msg=f"different results for {(fc_age, mp_age)} with epsilon: {epsilon}")

    def _compare_fc_mp_indexed(self, expected, obtained):
        expected_index_set = set(expected.index)
        obtained_index_set = set(obtained.index)
        self.assertTrue(obtained_index_set.issuperset(expected_index_set),
                        f"missing {expected_index_set.difference(obtained_index_set)}")

        for fc_value, mp_value in expected.index:
            expected_value = self._safe_cast_to_float(expected[fc_value][mp_value])
            obtained_value = obtained.loc[fc_value][mp_value]

            self.assertAlmostEqual(expected_value, obtained_value,
                                   places=8,
                                   msg=f"different results for {(fc_value, mp_value)}")

    def _safe_cast_to_float(self, value) -> float:
        if type(value) == str:
            return float(value.replace(',', '.').replace('E', 'e'))
        else:
            return value
