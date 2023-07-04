import pandas as pd
from pandas import DataFrame

from nonGenCom.Utils import FC_INDEX_NAME
from nonGenCom.Variables.AgeAbstract import AgeAbstract


class AgeContinuous(AgeAbstract):
    SCORE_COLNAME = 'age_v2_score'

    def __init__(self, context_name='Standard', min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv",
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(contexts_path, sigmas_path)

        self.min_age = min_age
        self.max_age = max_age

        category_ranges = {}
        for i in range(self.min_age, self.max_age):
            category_ranges[i] = (i, i)
        self.category_ranges = category_ranges

        self.version_name = 'V2'

        # in order to not re-calculate for all posterior calls
        self.likelihood = self.get_FC_likelihood()
        self.prior = self.get_context(context_name)
        self.evidence = self._calculate_evidence(self.prior, self.likelihood)
        self.posteriors = {}

        self.mp_scores = {}

    def get_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_age: int) -> float | None:
        if pd.isna(mp_age):
            return None

        from_age = fc_min_age if not pd.isna(fc_min_age) else self.min_age
        to_age = fc_max_age if not pd.isna(fc_max_age) else self.max_age

        key = (from_age, to_age)
        if key in self.posteriors:
            if mp_age in self.posteriors[key]:
                return self.posteriors[key][mp_age]

        fc_age_range = range(from_age, to_age + (1 if from_age == to_age else 0))

        l_filter = self.likelihood.index.get_level_values(0).isin(fc_age_range) & \
                   (self.likelihood.index.get_level_values(1) == str(mp_age))

        sum_likelihoods_x_prior = sum(self.likelihood.loc[l_filter]) * self.prior[mp_age]

        posterior = round(sum_likelihoods_x_prior / self._get_evidence_for_range(fc_age_range), self.DECIMAL_PRECISION)

        self.posteriors.setdefault(key, {})[mp_age] = posterior
        return posterior

    def _get_evidence_for_range(self, fc_age_range: range):
        e_filter = self.evidence.index.get_level_values(0).isin(fc_age_range)
        return sum(self.evidence.loc[e_filter])

    def add_score_fc_by_apply(self, merged_dbs: DataFrame, fc_min_age_colname: str, fc_max_age_colname: str,
                              mp_value_colname: str) -> DataFrame:

        merged_dbs[self.SCORE_COLNAME] = merged_dbs.apply(
            lambda row: self.get_posterior_for_case(row[fc_min_age_colname], row[fc_max_age_colname], row[mp_value_colname]), axis=1
        )

        merged_dbs = merged_dbs.reset_index(drop=True)\
            .sort_values(self.SCORE_COLNAME, ascending=False)

        return merged_dbs

    def add_score_mp_by_apply(self, merged_dbs: DataFrame, fc_min_age_colname: str, fc_max_age_colname: str,
                              mp_value_colname: str) -> DataFrame:

        merged_dbs[self.SCORE_COLNAME] = merged_dbs.apply(
            lambda row: self._get_mp_score(row[fc_min_age_colname], row[fc_max_age_colname], row[mp_value_colname]),
            axis=1
        )

        merged_dbs = merged_dbs.reset_index(drop=True)\
            .sort_values(self.SCORE_COLNAME, ascending=False)

        return merged_dbs

    def _get_mp_score(self, fc_min_age: int, fc_max_age: int, mp_age: int):
        if pd.isna(mp_age):
            return None

        from_age = fc_min_age if not pd.isna(fc_min_age) else self.min_age
        to_age = fc_max_age if not pd.isna(fc_max_age) else self.max_age

        key = (fc_min_age, fc_max_age)
        if key in self.mp_scores:
            if mp_age in self.mp_scores[key]:
                return self.mp_scores[key][mp_age]

        fc_age_range = range(from_age, to_age + (1 if from_age == to_age else 0))
        l_categories = self.likelihood.index.get_level_values(0).isin(fc_age_range)

        mp_score = sum(self.likelihood.loc[l_categories & mp_age])

        self.mp_scores.setdefault(key, {})[mp_age] = mp_score
        return mp_score
