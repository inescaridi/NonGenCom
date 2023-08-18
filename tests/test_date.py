import datetime
from unittest import TestCase

import pandas as pd

from nonGenCom.Variables.Date import Date


class TestDate(TestCase):

    def test_fc_likelihood(self):
        initial_date = datetime.date(2000, 1, 1)
        final_date = datetime.date(2000, 10, 26)
        delta_in_days = 30

        context_config = []
        for i in range(10):
            context_config.append((initial_date + datetime.timedelta(days=i * delta_in_days), 0.1))  # uniform

        date_var = Date(initial_date, final_date, delta_in_days, geometrical_q=0.5, context_config=context_config)

        expected = pd.read_csv(f"tests/resources/date_q05_10periods.csv", index_col=0)

        obtained = date_var.fc_likelihood
        obtained = obtained.unstack()

        for fc_period in date_var.value_range:
            for r_period in date_var.value_range:
                self.assertAlmostEqual(expected.iloc[fc_period, r_period], obtained.iloc[fc_period, r_period], places=8,
                                       msg=f"different results for {(fc_period, r_period)}")
