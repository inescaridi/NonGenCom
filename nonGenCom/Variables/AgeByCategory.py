from pandas import Series

from nonGenCom.Utils import load_fc_indexed_file, FC_INDEX_NAME, load_mp_indexed_file
from nonGenCom.Variables.CategoricalVariable import CategoricalVariable, Category
import statistics as st


class AgeByCategory(CategoricalVariable):
    SCORE_COLNAME = 'age_v1_score'

    def __init__(self, fc_category_ranges_path="nonGenCom/default_inputs/age_ranges.csv",
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv",
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv"):
        self.sigmas = load_mp_indexed_file(sigmas_path)
        self.sigmas.index = self.sigmas.index.astype(int)

        fc_ranges_df = load_fc_indexed_file(fc_category_ranges_path)
        fc_categories = fc_ranges_df.groupby(FC_INDEX_NAME).agg({'age': (min, max)})['age'].apply(tuple, axis=1).to_dict()
        fc_categories = Category(fc_categories)

        min_age = int(fc_ranges_df.age.min())
        max_age = int(fc_ranges_df.age.max())
        mp_categories = {age: (age, age) for age in range(min_age, max_age + 1)}
        mp_categories = Category(mp_categories)

        r_categories = mp_categories

        super().__init__(contexts_path, None, r_categories, mp_categories, fc_categories)

        self.version_name = 'V1'

    def _get_fc_likelihood_for_category(self, scenery_name: str, mp_category: tuple, fc_category: tuple):
        mp_age = mp_category[0]
        fc_category_name = fc_category[0]
        fc_category_min_age, fc_category_max_age = fc_category[1]

        sigma = float(self.sigmas.loc[mp_age].iloc[0])
        normal_distribution = st.NormalDist(mp_age, sigma)
        lower = normal_distribution.cdf(self.mp_categories.max_value) - normal_distribution.cdf(self.mp_categories.min_value)

        upper = normal_distribution.cdf(min(fc_category_max_age + 1, self.mp_categories.max_value)) - normal_distribution.cdf(fc_category_min_age)
        value = upper / lower
        return value

    def _reformat_prior(self, prior: Series):
        prior.index = prior.index.astype(int)
        return prior
