import datetime
from unittest import TestCase

import pandas as pd

from nonGenCom.Variables.Date import Date


class TestDate(TestCase):

    def test_fc_likelihood(self):
        initial_date = datetime.date(2023, 1, 10)
        final_date = datetime.date(2023, 10, 10)
        delta_in_days = 10

        date_var = Date(initial_date, final_date, delta_in_days)

        expected = pd.read_csv(f"tests/resources/date/Date_FC_likelihood.csv", index_col=0)

        obtained = date_var.get_fc_likelihood()
        obtained = obtained.unstack()

        for fc_period in date_var.value_range:
            for r_period in date_var.value_range:
                self.assertAlmostEqual(expected.iloc[fc_period, r_period], obtained.iloc[fc_period, r_period], places=8,
                                       msg=f"different results for {(fc_period, r_period)}")
