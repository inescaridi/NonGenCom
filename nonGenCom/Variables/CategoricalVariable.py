from abc import abstractmethod
from operator import itemgetter

import pandas as pd
from pandas import Series

from nonGenCom.Utils import MP_INDEX_NAME, FC_INDEX_NAME
from nonGenCom.Variables.Variable import Variable


class Category:
    def __init__(self, categories: dict[str | int | float, tuple]):
        self.categories = categories
        self.min_value = min(categories.values(), key=itemgetter(0))[0]
        self.max_value = max(categories.values(), key=itemgetter(1))[1]

    def items(self):
        return self.categories.items()


class CategoricalVariable(Variable):
    def __init__(self, contexts_path, sceneries_path, r_categories: Category, mp_categories: Category,
                 fc_categories: Category):
        super().__init__(contexts_path, sceneries_path)
        self.r_categories = r_categories
        self.mp_categories = mp_categories
        self.fc_categories = fc_categories

    def get_fc_likelihood(self, scenery_name: str = None) -> Series:
        if scenery_name:
            if scenery_name in self.sceneries:
                return self.sceneries[scenery_name]
            else:
                raise ValueError(f"Scenery {scenery_name} not found in sceneries file")

        likelihood_list = []
        for mp_category in self.mp_categories.items():
            for fc_category in self.fc_categories.items():
                value = self._get_fc_likelihood_for_category(scenery_name, mp_category, fc_category)
                likelihood_list.append({FC_INDEX_NAME: fc_category[0], MP_INDEX_NAME: mp_category[0],
                                        'likelihood': value})

        likelihood = pd.DataFrame(likelihood_list).set_index([FC_INDEX_NAME, MP_INDEX_NAME])['likelihood']
        return likelihood

    def get_fc_posterior(self, context_name: str, scenery_name: str = None) -> Series:
        prior = self.get_prior(context_name)
        likelihood = self.get_fc_likelihood(scenery_name)

        return self._calculate_bayes(prior, likelihood)

    def get_prior(self, context_name: str) -> Series:
        prior = self.get_context(context_name)
        prior = self._reformat_prior(prior)
        return prior

    @abstractmethod
    def _get_fc_likelihood_for_category(self, scenery_name: str, mp_category: tuple, fc_category: tuple):
        raise NotImplementedError()

    @abstractmethod
    def _reformat_prior(self, prior: Series):
        raise NotImplementedError()
