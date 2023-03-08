from optparse import Option
from typing import Any

import pandas as pd
from pandas import DataFrame

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME
from nonGenCom.Variables.Age import Age


class AgeV2(Age):
    SCORE_COLNAME = 'age_v2_score'

    def __init__(self, min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv", sceneries_path=None,
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(contexts_path, sceneries_path, sigmas_path)

        self.min_age = min_age
        self.max_age = max_age

        category_ranges = {}
        for i in range(self.min_age, self.max_age):
            category_ranges[i] = (i, i)
        self.category_ranges = category_ranges

        self.version_name = 'V2'

        # in order to not re-calculate for all posterior calls
        self.likelihood = self.get_likelihood()
        self.prior = None
        self.evidence = None
        self.posteriors = {}

        self.mp_scores = {}

    def set_context(self, context_name):
        self.prior = self.get_context(context_name)

    def get_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_age: int) -> float | None:
        if pd.isna(mp_age):
            return None

        key = (fc_min_age, fc_max_age)
        if key in self.posteriors:
            if mp_age in self.posteriors[key]:
                return self.posteriors[key][mp_age]

        from_age = fc_min_age if not pd.isna(fc_min_age) else self.min_age
        to_age = fc_max_age if not pd.isna(fc_max_age) else self.max_age

        fc_age_range = range(from_age, to_age)

        l_filter = self.likelihood.index.get_level_values(0).isin(fc_age_range) & \
                   (self.likelihood.index.get_level_values(1) == str(mp_age))

        sum_likelihoods_x_prior = sum(self.likelihood.loc[l_filter]) * self.prior[mp_age]

        posterior = round(sum_likelihoods_x_prior / self._get_evidence_for_range(fc_age_range), self.DECIMAL_PRECISION)

        self.posteriors.setdefault(key, {})[mp_age] = posterior
        return posterior

    def _get_evidence_for_range(self, fc_age_range: range):
        evidence = self.get_evidence()
        e_filter = evidence.index.get_level_values(0).isin(fc_age_range)
        return round(sum(evidence.loc[e_filter]), self.DECIMAL_PRECISION)

    def get_evidence(self):
        if self.evidence is None:
            if self.prior is None:
                print("WARNING context not setted")
            print("Calculating evidence")

            likelihood_x_prior = self.likelihood.multiply(self.prior, level=1)
            self.evidence = likelihood_x_prior.groupby(FC_INDEX_NAME).sum()
            print("Done")
        return self.evidence

    def add_score_fc_by_apply(self, merged_dbs: DataFrame, context_name: str, scenery_name: str,
                              fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        self.set_context(context_name)
        print(f"Context: {context_name}")
        print(f"Scenery: {scenery_name}")

        merged_dbs[self.SCORE_COLNAME] = merged_dbs.apply(
            lambda row: self.get_posterior_for_case(row[fc_value_colname], 0, row[mp_value_colname]), axis=1
        )

        merged_dbs = merged_dbs.reset_index(drop=True)\
            .sort_values(self.SCORE_COLNAME, ascending=False)

        return merged_dbs

    def add_score_mp_by_apply(self, merged_dbs: DataFrame, scenery_name: str,
                              fc_value_colname: str, mp_value_colname: str) -> DataFrame:

        merged_dbs[self.SCORE_COLNAME] = merged_dbs.apply(
            lambda row: self._get_mp_score(row[fc_value_colname], row[mp_value_colname]), axis=1
        )

        merged_dbs = merged_dbs.reset_index(drop=True)\
            .sort_values(self.SCORE_COLNAME, ascending=False)

        return merged_dbs

    def _get_mp_score(self, fc_category, mp_age):
        if pd.isna(fc_category) or pd.isna(mp_age):
            return None

        if fc_category in self.mp_scores:
            if mp_age in self.mp_scores[fc_category]:
                return self.mp_scores[fc_category][mp_age]

        min_v, max_v = self.category_ranges[fc_category]
        c_range = range(int(min_v), int(max_v) + 1)

        l_categories = self.likelihood.index.get_level_values(0).isin(c_range)

        mp_score = sum(self.likelihood.loc[l_categories & mp_age])

        self.mp_scores.setdefault(fc_category, {})[mp_age] = mp_score
        return mp_score
