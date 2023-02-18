import statistics as st
from typing import List, Tuple

import pandas as pd
from pandas import Series

from nonGenCom.Utils import load_fc_indexed_file, load_mp_indexed_file
from nonGenCom.Variable import Variable


class Age(Variable):
    def __init__(self, contexts_path="nonGenCom/default_inputs/age_contexts.csv",
                 sceneries_path=None):
        super().__init__(contexts_path, sceneries_path)
        self.sigmas = load_mp_indexed_file("nonGenCom/default_inputs/age_sigma.csv")

        ranges_df = load_fc_indexed_file("nonGenCom/default_inputs/age_ranges.csv")
        self.category_ranges = ranges_df.groupby('FC').agg({'age': (min, max)})['age'].apply(tuple, axis=1).to_dict()

    def get_posterior(self, context_name: str, scenery_name: str = None) -> Series:
        prior = self.get_context(context_name)
        if scenery_name is not None and scenery_name in self.sceneries:
            likelihood = self.get_scenery(scenery_name)
        else:
            min_age, max_age, category_ranges = self.get_category_ranges()
            likelihood = self.get_likelihood_v1(min_age, max_age, category_ranges)
            print("likelihood_v1\n", likelihood)  # TODO remove or use logger

        return self._calculate_bayes(prior, likelihood)

    def profiling(self, prior: Series, likelihood: Series, cos_pairs: List[str] = None, cow_pairs: List[str] = None,
                  ins_pairs: List[str] = None, inw_pairs: List[str] = None):
        pass

    def get_likelihood_v1(self, min_age: int, max_age: int, category_ranges: dict[str, Tuple[int, int]]) -> Series:
        """
        The method computes de conditional probability of a chosen range of ages given it was a particular age
        or more formally:   P(FC = category | MP = missing_person_age)
        :param min_age minimum possible age
        :param max_age maximum possible age
        :param category_ranges: dict: with category name as key and range as value
        :return:
        """

        likelihood = []
        for category_name, category_range in category_ranges.items():
            for mp_age in range(min_age, max_age+1):
                if str(mp_age) not in self.sigmas.index:
                    print(f"missing {mp_age} in sigma file!")
                sigma = float(self.sigmas.loc[str(mp_age)])

                normal_distribution = st.NormalDist(mp_age, sigma)
                lower = normal_distribution.cdf(max_age) - normal_distribution.cdf(min_age)

                category_min_age, category_max_age = category_range
                upper = normal_distribution.cdf(category_max_age) - normal_distribution.cdf(category_min_age)
                value = upper / lower
                likelihood.append({'FC': category_name, 'MP': str(mp_age), 'likelihood': value})

        return pd.DataFrame(likelihood).set_index(['FC', 'MP'])['likelihood']

    def get_likelihood_v2(self, min_age: int, max_age: int):
        category_ranges = {}
        for i in range(min_age, max_age - 1):
            category_ranges[f"Category_{i}"] = (i, i + 1)
        return self.get_likelihood_v1(min_age, max_age, category_ranges)

    def get_category_ranges(self) -> tuple[int, int, dict[str, Tuple[int, int]]]:
        sigmas_no_index = self.sigmas.reset_index()
        sigmas_no_index['MP'] = sigmas_no_index['MP'].astype(int)
        min_age, max_age = sigmas_no_index['MP'].min(), sigmas_no_index['MP'].max()
        return min_age, max_age, self.category_ranges
