from abc import ABC, abstractmethod

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
        self.mp_likelihood = self.get_mp_likelihood(mp_scenery_name)

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

    def get_fc_score(self) -> Series:
        evidence = self._calculate_evidence(self.prior, self.fc_likelihood)
        return self.score_numerator.divide(evidence, level=FC_INDEX_NAME)

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

    def get_mp_score(self) -> Series:
        evidence = self._calculate_evidence(self.prior, self.mp_likelihood)
        return self.score_numerator.divide(evidence, level=MP_INDEX_NAME)

    @abstractmethod
    def _get_fc_likelihood_for_combination(self, r_value: int | float, fc_value: int | float):
        raise NotImplementedError()

    @abstractmethod
    def _get_mp_likelihood_for_combination(self, r_value: int | float, mp_category: int | float):
        raise NotImplementedError()
