from abc import abstractmethod, ABC

import pandas as pd
from pandas import Series

from nonGenCom.Utils import MP_INDEX_NAME, FC_INDEX_NAME, R_INDEX_NAME
from nonGenCom.Variable import Variable


class CategoricalVariable(Variable, ABC):
    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None,
                 r_categories: dict, mp_categories: dict, fc_categories: dict):
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name)
        self.r_categories = r_categories
        self.mp_categories = mp_categories
        self.fc_categories = fc_categories

        self.prior = self.get_prior(context_name)
        self.fc_likelihood = self.get_fc_likelihood(fc_scenery_name)
        self.mp_likelihood = self.get_mp_likelihood(mp_scenery_name)

        self.score_numerator = self._get_score_numerator(self.fc_likelihood,
                                                         self.mp_likelihood,
                                                         self.prior,
                                                         self.fc_categories.items(),
                                                         self.mp_categories.items())

    def get_fc_likelihood(self, scenery_name: str = None) -> Series:
        # TODO see if we move part of this up
        scenery = self.get_fc_scenery(scenery_name)
        if scenery is not None:
            return scenery

        likelihood_list = []
        for fc_category in self.fc_categories.items():
            for r_category in self.r_categories.items():
                value = self._get_fc_likelihood_for_combination(r_category, fc_category)
                likelihood_list.append({FC_INDEX_NAME: fc_category[0], R_INDEX_NAME: r_category[0],
                                        'likelihood': value})

        likelihood = pd.DataFrame(likelihood_list).set_index([FC_INDEX_NAME, R_INDEX_NAME])['likelihood']
        return likelihood

    def get_fc_score(self) -> Series:
        evidence = self._calculate_evidence(self.fc_likelihood, self.prior)
        return self.score_numerator.divide(evidence, level=FC_INDEX_NAME)

    def get_mp_likelihood(self, scenery_name: str) -> Series:
        # TODO see if we move part of this up
        scenery = self.get_mp_scenery(scenery_name)
        if scenery is not None:
            return scenery

        likelihood_list = []
        for mp_category in self.mp_categories.items():
            for r_category in self.r_categories.items():
                value = self._get_mp_likelihood_for_combination(r_category, mp_category)
                likelihood_list.append({FC_INDEX_NAME: mp_category[0], R_INDEX_NAME: r_category[0],
                                        'likelihood': value})

        likelihood = pd.DataFrame(likelihood_list).set_index([FC_INDEX_NAME, R_INDEX_NAME])['likelihood']
        return likelihood

    def get_mp_score(self, context_name: str, scenery_name: str) -> Series:
        evidence = self._calculate_evidence(self.mp_likelihood, self.prior)
        return self.score_numerator.divide(evidence, level=MP_INDEX_NAME)

    def get_prior(self, context_name: str) -> Series:
        prior = self.get_context(context_name)
        prior = self._reformat_prior(prior)
        return prior

    @abstractmethod
    def _get_fc_likelihood_for_combination(self, r_category: tuple, fc_category: tuple):
        raise NotImplementedError()

    @abstractmethod
    def _get_mp_likelihood_for_combination(self, r_category: tuple, mp_category: tuple):
        raise NotImplementedError()
