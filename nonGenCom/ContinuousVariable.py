from abc import ABC, abstractmethod
from functools import lru_cache

import pandas as pd
from pandas import Series

from nonGenCom.Utils import FC_INDEX_NAME, R_INDEX_NAME, MP_INDEX_NAME
from nonGenCom.Variable import Variable


class ContinuousVariable(Variable, ABC):
    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None,
                 min_value, max_value, step):
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name)
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.value_range = range(self.min_value, self.max_value, self.step)

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

    def get_fc_likelihood(self, scenery_name: str) -> Series:
        # TODO see if we move part of this up
        scenery = self.get_fc_scenery(scenery_name)
        if scenery is not None:
            return scenery

        likelihood_list = []
        for fc_value in self.value_range:
            for r_value in self.value_range:
                likelihood_value = self._get_fc_likelihood_for_combination(r_value, fc_value)
                likelihood_list.append({FC_INDEX_NAME: fc_value, R_INDEX_NAME: r_value,
                                        'likelihood': likelihood_value})

        likelihood = pd.DataFrame(likelihood_list).set_index([FC_INDEX_NAME, R_INDEX_NAME])['likelihood']
        return likelihood

    def get_mp_likelihood(self, scenery_name: str) -> Series:
        # TODO see if we move part of this up
        scenery = self.get_mp_scenery(scenery_name)
        if scenery is not None:
            return scenery

        likelihood_list = []
        for mp_value in self.value_range:
            for r_value in self.value_range:
                likelihood_value = self._get_mp_likelihood_for_combination(r_value, mp_value)
                likelihood_list.append({MP_INDEX_NAME: mp_value, R_INDEX_NAME: r_value,
                                        'likelihood': likelihood_value})

        likelihood = pd.DataFrame(likelihood_list).set_index([MP_INDEX_NAME, R_INDEX_NAME])['likelihood']
        return likelihood

    @lru_cache(maxsize=128)
    def get_fc_score_for_range(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int) -> Series:
        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))
        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))

        filter_age_range = self.score_numerator.index.get_level_values(0).isin(fc_age_range) & \
                           self.score_numerator.index.get_level_values(1).isin(mp_age_range)

        posterior_nominator = self.score_numerator.loc[filter_age_range].sum().item()
        fc_posterior_denominator = self.fc_evidence.loc[fc_age_range].sum() * len(mp_age_range)

        return posterior_nominator / fc_posterior_denominator

    @lru_cache(maxsize=128)
    def get_mp_score_for_range(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int) -> Series:
        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))
        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))

        filter_age_range = self.score_numerator.index.get_level_values(0).isin(fc_age_range) & \
                           self.score_numerator.index.get_level_values(1).isin(mp_age_range)

        posterior_numerator = self.score_numerator.loc[filter_age_range].sum().item()
        mp_posterior_denominator = self.mp_evidence.loc[mp_age_range].sum() * len(fc_age_range)

        return posterior_numerator / mp_posterior_denominator

    @lru_cache(maxsize=128)
    def get_fc_mp_score_for_range(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int):
        fc_posterior = self.get_fc_score_for_range(fc_min_age, fc_max_age, mp_min_age, mp_max_age)
        mp_posterior = self.get_mp_score_for_range(fc_min_age, fc_max_age, mp_min_age, mp_max_age)
        return fc_posterior, mp_posterior

    @abstractmethod
    def _get_fc_likelihood_for_combination(self, r_value: int | float, fc_value: int | float):
        raise NotImplementedError()

    @abstractmethod
    def _get_mp_likelihood_for_combination(self, r_value: int | float, mp_value: int | float):
        raise NotImplementedError()
