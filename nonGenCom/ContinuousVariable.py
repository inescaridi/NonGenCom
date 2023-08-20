from abc import ABC, abstractmethod
from functools import lru_cache

from pandas import Series, DataFrame

from nonGenCom.Variable import Variable


class ContinuousVariable(Variable, ABC):
    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None,
                 min_value, max_value, step):
        """

        :param contexts_path:
        :param fc_sceneries_path:
        :param mp_sceneries_path:
        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        :param min_value:
        :param max_value:
        :param step:
        """
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.value_range = range(min_value, max_value+1, step)

        self.prior = self.get_prior(context_name)

        self.fc_likelihood = self.get_fc_likelihood(fc_scenery_name)
        self.fc_evidence = self._calculate_evidence(self.prior, self.fc_likelihood)

        self.mp_likelihood = self.get_mp_likelihood(mp_scenery_name)
        self.mp_evidence = self._calculate_evidence(self.prior, self.mp_likelihood)

        self.score_numerator = self._get_score_numerator(self.fc_likelihood,
                                                         self.mp_likelihood,
                                                         self.prior,
                                                         self.value_range,
                                                         self.value_range)

    def get_fc_likelihood(self, scenery_name: str = None) -> Series:
        return self._calculate_fc_likelihood(scenery_name, self.value_range, self.value_range)

    def get_mp_likelihood(self, scenery_name: str = None) -> Series:
        return self._calculate_mp_likelihood(scenery_name, self.value_range, self.value_range)

    @lru_cache(maxsize=128)
    def _calculate_fc_score_for_range(self, fc_min_value: int, fc_max_value: int, mp_min_value: int, mp_max_value: int) -> Series:
        """
        Internal function to calculate the fc score, given a range of fc and mp values, they must be integers
        :param fc_min_value: int:
        :param fc_max_value: int:
        :param mp_min_value: int:
        :param mp_max_value: int:
        :return:
        """
        fc_range = range(fc_min_value, fc_max_value + (1 if fc_min_value == fc_max_value else 0))
        mp_range = range(mp_min_value, mp_max_value + (1 if mp_min_value == mp_max_value else 0))

        filter_range = self.score_numerator.index.get_level_values(0).isin(fc_range) & \
                           self.score_numerator.index.get_level_values(1).isin(mp_range)

        posterior_nominator = self.score_numerator.loc[filter_range].sum().item()
        fc_posterior_denominator = self.fc_evidence.loc[fc_range].sum() * len(mp_range)

        return posterior_nominator / fc_posterior_denominator

    @lru_cache(maxsize=128)
    def _calculate_mp_score_for_range(self, fc_min_value: int, fc_max_value: int, mp_min_value: int, mp_max_value: int) -> Series:
        """
        Internal function to calculate the mp score, given a range of fc and mp values, they must be integers
        :param fc_min_value: int:
        :param fc_max_value: int:
        :param mp_min_value: int:
        :param mp_max_value: int:
        :return:
        """
        fc_range = range(fc_min_value, fc_max_value + (1 if fc_min_value == fc_max_value else 0))
        mp_range = range(mp_min_value, mp_max_value + (1 if mp_min_value == mp_max_value else 0))

        filter_range = self.score_numerator.index.get_level_values(0).isin(fc_range) & \
                           self.score_numerator.index.get_level_values(1).isin(mp_range)

        posterior_numerator = self.score_numerator.loc[filter_range].sum().item()
        mp_posterior_denominator = self.mp_evidence.loc[mp_range].sum() * len(fc_range)

        return posterior_numerator / mp_posterior_denominator

    def add_fc_score(self, merged_dbs: DataFrame, fc_min_value_colname: str, fc_max_value_colname: str,
                     mp_min_value_colname: str, mp_max_value_colname: str) -> DataFrame:
        score_colname = self.score_colname_template.format('fc')
        merged_dbs[score_colname] = merged_dbs.apply(
            lambda row: self.get_fc_score_for_range(row[fc_min_value_colname], row[fc_max_value_colname],
                                                    row[mp_min_value_colname], row[mp_max_value_colname]), axis=1)

        merged_dbs = merged_dbs.reset_index(drop=True)\
            .sort_values(score_colname, ascending=False)

        return merged_dbs

    def add_mp_score(self, merged_dbs: DataFrame, fc_min_value_colname: str, fc_max_value_colname: str,
                     mp_min_value_colname: str, mp_max_value_colname: str) -> DataFrame:
        score_colname = self.score_colname_template.format('mp')
        merged_dbs[score_colname] = merged_dbs.apply(
            lambda row: self.get_mp_score_for_range(row[fc_min_value_colname], row[fc_max_value_colname],
                                                    row[mp_min_value_colname], row[mp_max_value_colname]), axis=1)

        merged_dbs = merged_dbs.reset_index(drop=True)\
            .sort_values(score_colname, ascending=False)

        return merged_dbs

    @abstractmethod
    def get_fc_score_for_range(self, fc_min_value, fc_max_value, mp_min_value, mp_max_value) -> Series:
        """
        Returns the fc score for a given range of values, its types depend on the concrete variable implementation (ex. Date).
        """
        raise NotImplementedError()

    @abstractmethod
    def get_mp_score_for_range(self, fc_min_value, fc_max_value, mp_min_value, mp_max_value) -> Series:
        """
        Returns the mp score for a given range of values, its types depend on the concrete variable implementation (ex. Date).
        """
        raise NotImplementedError()
