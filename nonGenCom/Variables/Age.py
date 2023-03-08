import statistics as st
from typing import List

import pandas as pd
from pandas import Series

from nonGenCom.Utils import load_mp_indexed_file, load_fc_indexed_file, FC_INDEX_NAME, MP_INDEX_NAME
from nonGenCom.Variables.Variable import Variable


class Age(Variable):
    SCORE_COLNAME = 'SHOULD USE AGE V1 OR V2'

    def __init__(self, contexts_path="nonGenCom/default_inputs/age_contexts.csv", sceneries_path=None,
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv",
                 category_ranges_path="nonGenCom/default_inputs/age_ranges.csv"):
        super().__init__(contexts_path, sceneries_path)
        self.sigmas = load_mp_indexed_file(sigmas_path)

        # default values
        self.category_ranges = {}
        self.min_age: int = -1
        self.max_age: int = 100
        self.version_name = 'BASE'

    def get_posterior(self, context_name: str, scenery_name: str = None) -> Series:
        prior = self.get_context(context_name)
        if scenery_name is not None and scenery_name != '' and scenery_name in self.sceneries:
            likelihood = self.get_scenery(scenery_name)
        else:
            likelihood = self.get_likelihood()

        return self._calculate_bayes(prior, likelihood)

    def profiling(self, prior: Series, likelihood: Series, cos_pairs: List[str] = None, cow_pairs: List[str] = None,
                  ins_pairs: List[str] = None, inw_pairs: List[str] = None):
        pass

    def get_likelihood(self, scenery_name=None) -> Series:
        """
        The method computes de conditional probability of a chosen range of ages given it was a particular age
        or more formally:   P(FC = category | MP = missing_person_age)
        :return:
        """

        likelihood_list = []
        for mp_age in range(self.min_age, self.max_age + 1):
            if str(mp_age) not in self.sigmas.index:
                print(f"missing {mp_age} in sigma file!")

            sigma = float(self.sigmas.loc[str(mp_age)])
            normal_distribution = st.NormalDist(mp_age, sigma)
            lower = normal_distribution.cdf(self.max_age) - normal_distribution.cdf(self.min_age)

            for category_name, category_range in self.category_ranges.items():
                category_min_age, category_max_age = category_range

                upper = normal_distribution.cdf(min(category_max_age+1, self.max_age)) - normal_distribution.cdf(category_min_age)
                value = upper / lower

                likelihood_list.append({FC_INDEX_NAME: category_name, MP_INDEX_NAME: str(mp_age), 'likelihood': value})

        likelihood = pd.DataFrame(likelihood_list).set_index([FC_INDEX_NAME, MP_INDEX_NAME])['likelihood']
        # print(f"Age_{self.version_name}\n", likelihood_list)  # TODO remove or use logger

        return likelihood.astype(float).round(self.DECIMAL_PRECISION)
