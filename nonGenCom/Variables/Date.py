import datetime
from math import floor

import pandas as pd
from pandas import Series

from nonGenCom.ContinuousVariable import ContinuousVariable
from nonGenCom.Utils import load_r_indexed_file, R_INDEX_NAME


class Date(ContinuousVariable):
    def __init__(self, initial_date: datetime.date | str, final_date: datetime.date | str, delta_in_days: int,
                 geometrical_q: float = 0.5,
                 context_config_filename="nonGenCom/scenery_and_context_inputs/date_context_config.csv",
                 context_config_name='Standard', context_config: list = None):
        """

        :param context_config_filename:
        :param initial_date: If a str is provided it should have the format %Y-%m-%d
        :param final_date:
        :param delta_in_days:
        :param geometrical_q:
        :param context_config_filename: filename of the context configuration file, should be R indexed with dates in
        format YYYY-MM-DD, with the accumulative probabilities for them, so each row is a different configuration
        :param context_config_name: name of the context to use, should be included in the context configuration file
        :param context_config: optional parameter to define the context configuration, it must be a list of tuples
        (date, probability) with the date as string in format YYYY-MM-DD.
        If this is provided the context_config_name is ignored
        """
        contexts_path = context_name = None
        fc_sceneries_path = fc_scenery_name = None
        mp_sceneries_path = mp_scenery_name = None

        self.initial_time = self._convert_to_date(initial_date)
        self.final_time = self._convert_to_date(final_date)
        self.delta_in_days = delta_in_days

        self.periods_date = pd.date_range(start=self.initial_time, end=self.final_time, freq=f"{self.delta_in_days}D")
        self.geometrical_q = geometrical_q

        if context_config is not None:  # TODO improve the way we configure the prior
            self.prior_definition = pd.DataFrame(context_config, columns=[R_INDEX_NAME, 'probability'])
        else:
            self.prior_definition = load_r_indexed_file(context_config_filename)[context_config_name].reset_index()

        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, 0, len(self.periods_date) - 1, 1)

    @staticmethod
    def _convert_to_date(date_to_convert) -> datetime.date:
        if isinstance(date_to_convert, str):
            return datetime.datetime.strptime(date_to_convert, "%Y-%m-%d").date()
        if isinstance(date_to_convert, datetime.datetime):
            return date_to_convert.date()
        if isinstance(date_to_convert, datetime.date):
            return date_to_convert
        else:
            raise ValueError()  # TODO add info to the exception

    def score_colname_template(self) -> str:
        return "date_{}_score"

    def _get_period_for_date(self, d: datetime.date | str):
        """

        :param d:
        :return:
        """
        d = self._convert_to_date(d)
        return floor((d - self.initial_time).days / self.delta_in_days)

    def _reformat_prior(self, prior: Series):
        config = self.prior_definition.copy()
        config[R_INDEX_NAME] = pd.to_datetime(config[R_INDEX_NAME], format="%Y-%m-%d").dt.date.apply(self._get_period_for_date)
        config.dropna().drop_duplicates(subset=[R_INDEX_NAME], inplace=True, keep='last')

        return config.set_index(R_INDEX_NAME).iloc[:, 0]

    def _get_fc_likelihood_for_combination(self, r_value, fc_value):
        return self.geometrical_q * ((1 - self.geometrical_q) ** (fc_value - r_value)) if fc_value >= r_value else 0

    def _get_mp_likelihood_for_combination(self, r_value, mp_value):
        return int(r_value == mp_value)  # we assume perfect representation for MP

    def get_fc_score_for_range(self, fc_min_value: datetime.date | str, fc_max_value: datetime.date | str,
                               mp_min_value: datetime.date | str, mp_max_value: datetime.date | str) -> Series:
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

    def get_mp_score_for_range(self, fc_min_value: datetime.date | str, fc_max_value: datetime.date | str,
                               mp_min_value: datetime.date | str, mp_max_value: datetime.date | str) -> Series:
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
