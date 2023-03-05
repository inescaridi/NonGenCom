import pandas as pd

from nonGenCom.Age import Age


class AgeV2(Age):
    def __init__(self, min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv",
                 sceneries_path=None,
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(contexts_path, sceneries_path, sigmas_path)

        self.min_age = min_age
        self.max_age = max_age

        category_ranges = {}
        for i in range(self.min_age, self.max_age - 1):
            category_ranges[i] = (i, i + 1)
        self.category_ranges_for_likelihood = category_ranges

        self.version_name = 'V2'

        # in order to not re-calculate likelihood for all posterior calls
        self.likelihood = self.get_likelihood()
        self.prior = None
        self.evidence = None

    def set_context(self, context_name):
        self.prior = self.get_context(context_name)

    def get_posterior_for_case(self, fc_category: str, mp_age: int) -> float:
        if self.evidence is None:
            if self.prior is None:
                print("WARNING context not setted")
            likelihood_x_prior = self.likelihood.multiply(self.prior, level=1)
            self.evidence = likelihood_x_prior.groupby('FC').sum()

            # change FC index to int
            idx = self.likelihood.index
            self.likelihood.index = self.likelihood.index.set_levels(idx.levels[0].astype(int), level=0)

        min_v, max_v = self.category_ranges[fc_category]
        c_range = range(int(min_v), int(max_v) + 1)

        l_categories = self.likelihood.index.get_level_values(0).isin(c_range)
        e_categories = self.evidence.index.get_level_values(0).isin(c_range)

        sum_likelihoods_x_prior = sum(self.likelihood.multiply(self.prior[mp_age], level=1).loc[l_categories & mp_age])

        posterior = round(sum_likelihoods_x_prior / sum(self.evidence.loc[e_categories]), self.DECIMAL_PRECISION)
        return posterior
