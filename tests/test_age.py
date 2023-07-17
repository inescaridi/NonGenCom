from unittest import TestCase

import pandas as pd

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME, load_fc_indexed_file
from nonGenCom.Variables.AgeByCategory import AgeByCategory
from nonGenCom.Variables.AgeContinuous import AgeContinuous
from nonGenCom.Variables.AgeMPRange import AgeMPRange


class TestAge(TestCase):

    def test_likelihood_v1(self):
        age_var = AgeByCategory()

        expected = pd.read_csv("tests/resources/age/age_likelihood_v1.csv")\
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME})\
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['likelihood']
        obtained = age_var.get_FC_likelihood()

        self._compare_fc_mp_indexed(expected, obtained)

    def test_FC_evidence_v1(self):
        age_v1 = AgeByCategory()

        expected = pd.read_csv("tests/resources/age/age_evidence_v1.csv").set_index(FC_INDEX_NAME)['Evidence']

        prior = age_v1.get_context('Standard')
        likelihood = age_v1.get_FC_likelihood()
        obtained = age_v1._calculate_evidence(prior, likelihood)

        for fc_value in expected.index:
            self.assertAlmostEqual(expected.loc[fc_value], obtained.loc[fc_value],
                                   places=8,
                                   msg=f"different results for {fc_value}")

    def test_posterior_v1(self):
        age_v1 = AgeByCategory()

        expected = pd.read_csv("tests/resources/age/age_posterior_v1.csv")\
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME})\
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['posterior']
        obtained = age_v1.get_posterior("Standard")

        self._compare_fc_mp_indexed(expected, obtained)

    def test_posterior_v2(self):
        ranges_df = load_fc_indexed_file("nonGenCom/default_inputs/age_ranges.csv")
        default_category_ranges = ranges_df.groupby(FC_INDEX_NAME).agg({'age': (min, max)})['age'].astype(int)\
            .apply(tuple, axis=1).to_dict()

        expected = pd.read_csv("tests/resources/age/age_posterior_v1.csv") \
            .rename(columns={'FC': FC_INDEX_NAME, 'MP': MP_INDEX_NAME}) \
            .set_index([FC_INDEX_NAME, MP_INDEX_NAME])['posterior'].astype(float)

        age_v2 = AgeContinuous(context_name='Standard')

        for category, age_range in default_category_ranges.items():
            age_min, age_max = age_range

            for mp_age in range(age_v2.min_age, age_v2.max_age+1):
                posterior = age_v2.get_posterior_for_case(age_min, age_max, mp_age)
                self.assertAlmostEqual(expected[category][mp_age], posterior, places=8,
                                       msg=f"different results for {(category, mp_age)}")

    def test_FC_evidence_v2(self):
        age_v2 = AgeContinuous(context_name='Standard')

        expected = pd.read_csv("tests/resources/age/age_evidence_v2.csv").set_index(FC_INDEX_NAME)['Evidence']
        obtained = age_v2.evidence

        for fc_value in expected.index:
            self.assertAlmostEqual(expected.loc[fc_value], obtained.loc[fc_value],
                                   places=8,
                                   msg=f"different results for {fc_value}")

    def test_likelihood_v2(self):
        expected = pd.read_csv("tests/resources/age/Age_FC_likelihood_v2.csv", index_col=0).stack()
        min_age = int(min(expected.index.levels[0].min(), expected.index.levels[1].astype(int).min()))
        max_age = int(max(expected.index.levels[0].max(), expected.index.levels[1].astype(int).max()))
        expected.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]

        age_v2 = AgeContinuous(context_name='Standard', min_age=min_age, max_age=max_age)

        obtained = age_v2.get_FC_likelihood()

        self._compare_fc_mp_indexed(expected, obtained)

    def test_MP_likelihood_v3(self):
        age_v3 = AgeMPRange()

        for epsilon in [0, 1, 2, 5, 10]:
            expected = pd.read_csv(f"tests/resources/age/Age_MP_likelihood_epsilon{epsilon}.csv", index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())
            obtained = age_v3.get_MP_likelihood(epsilon=epsilon, min_age=min_age, max_age=max_age)

            obtained = obtained.unstack()

            for mp_age in range(min_age, max_age+1):
                for r_age in range(min_age, max_age+1):
                    self.assertAlmostEqual(expected.iloc[mp_age, r_age], obtained.iloc[mp_age, r_age], places=8,
                                           msg=f"different results for {(mp_age, r_age)} with epsilon: {epsilon}")

    def test_MP_evidence_v3(self):
        for epsilon in [0, 1]:
            age_v3 = AgeMPRange(epsilon=epsilon)

            expected = pd.read_csv(f"tests/resources/age/Age_MP_evidence_epsilon{epsilon}.csv", index_col=0)['Evi']
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())

            obtained = age_v3.mp_evidence

            for age in range(min_age, max_age + 1):
                self.assertAlmostEqual(expected.loc[age], obtained.loc[age], places=8,
                                       msg=f"different results for {age} with epsilon: {epsilon}")

    def test_FC_evidence_v3(self):
        age_v3 = AgeMPRange()
        expected = pd.read_csv("tests/resources/age/Age_FC_evidence.csv", index_col=0)['Evi']
        min_age = int(expected.index.min())
        max_age = int(expected.index.max())

        obtained = age_v3.fc_evidence

        for age in range(min_age, max_age + 1):
            self.assertAlmostEqual(expected.loc[age], obtained.loc[age], places=8,
                                   msg=f"different results for {age}")

    def test_MP_posterior_v3(self):
        for epsilon in [0, 1]:
            age_v3 = AgeMPRange(epsilon=epsilon)

            expected = pd.read_csv(f"tests/resources/age/Age_MP_posterior_epsilon{epsilon}.csv", index_col=0)
            min_age = int(expected.index.min())
            max_age = int(expected.index.max())

            obtained = age_v3.mp_posterior
            obtained = obtained.unstack()

            for mp_age in range(min_age, max_age+1):
                for r_age in range(min_age, max_age+1):
                    self.assertAlmostEqual(expected.iloc[mp_age, r_age], obtained.iloc[mp_age, r_age], places=8,
                                           msg=f"different results for {(mp_age, r_age)} with epsilon: {epsilon}")

    def test_FC_posterior_v3(self):
        age_v3 = AgeMPRange()
        expected = pd.read_csv("tests/resources/age/Age_FC_posterior.csv", index_col=0)
        min_age = int(expected.index.min())
        max_age = int(expected.index.max())

        obtained = age_v3.fc_posterior
        obtained = obtained.unstack()

        for mp_age in range(min_age, max_age+1):
            for r_age in range(min_age, max_age+1):
                self.assertAlmostEqual(expected.iloc[mp_age, r_age], obtained.iloc[mp_age, r_age], places=8,
                                       msg=f"different results for {(mp_age, r_age)}")

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
