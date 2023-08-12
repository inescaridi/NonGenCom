from unittest import TestCase

import pandas as pd

from nonGenCom.Utils import FC_INDEX_NAME, R_INDEX_NAME, MP_INDEX_NAME
from nonGenCom.Variables.Body import Body


class TestBody(TestCase):

    def test_score_numerator(self):
        body = Body("Uniform", "Head & Neck/Disease", "Head & Neck/Disease", "Head & Neck/Disease")

        expected = pd.read_csv("tests/resources/body/Body_score_numerator_PriorUniform.csv", index_col=0).stack()
        expected.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]
        obtained = body.score_numerator

        for fc_value, r_value in expected.index:
            expected_value = expected.loc[fc_value][r_value]
            obtained_value = obtained.loc[fc_value][r_value]

            self.assertAlmostEqual(expected_value, obtained_value,
                                   places=8,
                                   msg=f"different results for {(fc_value, r_value)}")

    def test_fc_score(self):
        body = Body("Uniform", "Head & Neck/Disease", "Head & Neck/Disease", "Head & Neck/Disease")

        expected = pd.read_csv("tests/resources/body/Body_FCscore_PriorUniform.csv", index_col=0).stack()
        expected.index.names = [FC_INDEX_NAME, R_INDEX_NAME]
        obtained = body.get_fc_score()

        for fc_value, r_value in expected.index:
            expected_value = expected.loc[fc_value][r_value]
            obtained_value = obtained.loc[fc_value][r_value]

            self.assertAlmostEqual(expected_value, obtained_value,
                                   places=8,
                                   msg=f"different results for {(fc_value, r_value)}")

    def test_mp_score(self):
        body = Body("Uniform", "Head & Neck/Disease", "Head & Neck/Disease", "Head & Neck/Disease")

        expected = pd.read_csv("tests/resources/body/Body_MPscore_PriorUniform.csv", index_col=0).stack()
        expected.index.names = [MP_INDEX_NAME, R_INDEX_NAME]
        obtained = body.get_mp_score()

        for fc_value, r_value in expected.index:
            expected_value = expected.loc[fc_value][r_value]
            obtained_value = obtained.loc[fc_value][r_value]

            self.assertAlmostEqual(expected_value, obtained_value,
                                   places=8,
                                   msg=f"different results for {(fc_value, r_value)}")
