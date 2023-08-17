import datetime
from math import floor

import pandas as pd
from pandas import Series

from nonGenCom.ContinuousVariable import ContinuousVariable
from nonGenCom.Utils import load_r_indexed_file, R_INDEX_NAME


class Date(ContinuousVariable):
    def __init__(self, initial_date: datetime.date, final_date: datetime.date, delta_in_days: int,
                 geometrical_q: float = 0.5, context_name='Standard'):
        """

        :param initial_date:
        :param final_date:
        :param delta_in_days:
        :param geometrical_q:
        :param context_name:
        """
        contexts_path = None
        fc_sceneries_path = fc_scenery_name = None
        mp_sceneries_path = mp_scenery_name = None

        self.initial_time = initial_date
        self.final_time = final_date
        self.delta_in_days = delta_in_days

        self.periods_date = pd.date_range(start=self.initial_time, end=self.final_time, freq=f"{self.delta_in_days}D")
        self.geometrical_q = geometrical_q
        self.prior_definition = load_r_indexed_file("nonGenCom/scenery_and_context_inputs/date_context_config.csv")[context_name]

        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, 0, len(self.periods_date)-1, 1)

    def score_colname(self) -> str:
        return "date_score"

    def _get_period_for_date(self, d: datetime.date):
        """

        :param d:
        :return:
        """
        return floor((d - self.initial_time).days / self.delta_in_days)

    def _reformat_prior(self, prior: Series):
        config = self.prior_definition.reset_index()
        config['date'] = pd.to_datetime(config['R_i'], format="%Y-%m-%d").dt.date
        config[R_INDEX_NAME] = config['date'].apply(self._get_period_for_date)
        config.drop(columns=['date'], inplace=True)

        return config.set_index('R_i')[self.context_name]

    def _get_fc_likelihood_for_combination(self, r_category, fc_category):
        return self.geometrical_q * ((1 - self.geometrical_q) ** abs(fc_category - r_category))

    def _get_mp_likelihood_for_combination(self, r_category, mp_category):
        return int(r_category == mp_category)  # we assume perfect representation for MP

    def get_fc_score_for_range(self, fc_min_value: datetime.date, fc_max_value: datetime.date,
                               mp_min_value: datetime.date, mp_max_value: datetime.date) -> Series:
        """
        Returns the fc score for a given range of values
        :param fc_min_value: datetime.date:
        :param fc_max_value: datetime.date:
        :param mp_min_value: datetime.date:
        :param mp_max_value: datetime.date:
        :return:
        """
        fc_min_period = self._get_period_for_date(fc_min_value)
        fc_max_period = self._get_period_for_date(fc_max_value)
        mp_min_period = self._get_period_for_date(mp_min_value)
        mp_max_period = self._get_period_for_date(mp_max_value)
        return self._calculate_fc_score_for_range(fc_min_period, fc_max_period, mp_min_period, mp_max_period)

    def get_mp_score_for_range(self, fc_min_value: datetime.date, fc_max_value: datetime.date,
                               mp_min_value: datetime.date, mp_max_value: datetime.date) -> Series:
        """
        Returns the mp score for a given range of values
        :param fc_min_value: datetime.date:
        :param fc_max_value: datetime.date:
        :param mp_min_value: datetime.date:
        :param mp_max_value: datetime.date:
        :return:
        """
        fc_min_period = self._get_period_for_date(fc_min_value)
        fc_max_period = self._get_period_for_date(fc_max_value)
        mp_min_period = self._get_period_for_date(mp_min_value)
        mp_max_period = self._get_period_for_date(mp_max_value)
        return self._calculate_mp_score_for_range(fc_min_period, fc_max_period, mp_min_period, mp_max_period)
