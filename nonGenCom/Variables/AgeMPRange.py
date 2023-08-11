import os

import pandas as pd
from pandas import Series, DataFrame

from nonGenCom.Utils import MP_INDEX_NAME, R_INDEX_NAME, FC_INDEX_NAME, change_index_level_type
from nonGenCom.Variables.AgeContinuous import AgeContinuous


class AgeMPRange(AgeContinuous):
    SCORE_COLNAME = 'age_v3_score'

    def __init__(self, context_name='Standard', epsilon=1, min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/scenery_and_context_inputs/age_contexts.csv",
                 sigmas_path="nonGenCom/scenery_and_context_inputs/age_sigma.csv"):
        super().__init__(context_name, min_age, max_age, contexts_path, sigmas_path)

        self.epsilon = epsilon

        self.fc_likelihood = self.likelihood
        self.fc_likelihood.index.names = [FC_INDEX_NAME, R_INDEX_NAME]
        self.fc_likelihood = change_index_level_type(self.fc_likelihood, FC_INDEX_NAME, int)
        self.fc_evidence = self.evidence

        self.mp_prior = self.prior.copy()
        self.mp_prior.index.names = [R_INDEX_NAME]
        self.mp_prior.index = self.mp_prior.index.astype(int)
        self.mp_likelihood = self.get_mp_likelihood(epsilon, min_age, max_age)
        self.mp_evidence = self._calculate_evidence(self.mp_prior, self.mp_likelihood)

        # Posteriors
        self.score_numerator = self._get_score_merator()
        # posterior_numerator = self.fc_likelihood.mul(self.mp_likelihood.mul(self.mp_prior, level=1).groupby(level=1).sum(), level=1) # may be another way to do this
        self._fc_posteriors = {}
        self._mp_posteriors = {}

    def get_fc_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int):
        if any(pd.isna([fc_min_age, fc_max_age, mp_min_age, mp_max_age])):
            return None

        fc_key = (fc_min_age, fc_max_age)
        mp_key = (mp_min_age, mp_max_age)
        if fc_key in self._fc_posteriors:
            if mp_key in self._fc_posteriors[fc_key]:
                return self._fc_posteriors[fc_key][mp_key]

        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))
        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))

        filter_age_range = self.score_numerator.index.get_level_values(0).isin(fc_age_range) & \
                           self.score_numerator.index.get_level_values(1).isin(mp_age_range)

        posterior_nominator = self.score_numerator.loc[filter_age_range].sum().item()
        fc_posterior_denominator = self.fc_evidence.loc[fc_age_range].sum() * len(mp_age_range)

        fc_posterior_value = posterior_nominator / fc_posterior_denominator
        self._fc_posteriors.setdefault(fc_key, {})[mp_key] = fc_posterior_value

        return fc_posterior_value

    def get_fc_mp_posteriors_for_case(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int):
        fc_posterior = self.get_fc_posterior_for_case(fc_min_age, fc_max_age, mp_min_age, mp_max_age)
        mp_posterior = self.get_mp_posterior_for_case(fc_min_age, fc_max_age, mp_min_age, mp_max_age)
        return fc_posterior, mp_posterior

    def get_mp_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int):
        if any(pd.isna([fc_min_age, fc_max_age, mp_min_age, mp_max_age])):
            return None

        fc_key = (fc_min_age, fc_max_age)
        mp_key = (mp_min_age, mp_max_age)
        if mp_key in self._mp_posteriors:
            if fc_key in self._mp_posteriors[mp_key]:
                return self._mp_posteriors[mp_key][fc_key]

        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))
        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))

        filter_age_range = self.score_numerator.index.get_level_values(0).isin(fc_age_range) & \
                           self.score_numerator.index.get_level_values(1).isin(mp_age_range)

        posterior_numerator = self.score_numerator.loc[filter_age_range].sum().item()
        mp_posterior_denominator = self.mp_evidence.loc[mp_age_range].sum() * len(fc_age_range)

        mp_posterior_value = posterior_numerator / mp_posterior_denominator
        self._mp_posteriors.setdefault(mp_key, {})[fc_key] = mp_posterior_value

        return mp_posterior_value

    # TODO move method to abstract class Variable if it makes sense
    def get_mp_likelihood(self, epsilon: int = 1, min_age: int = -1, max_age: int = 100) -> Series:
        """
        Calculates the MP likelihood, we assume a uniform probability distribution over the interval [z-e, z+e] for
        the reported A_MP value
        :param epsilon:
        :param min_age:
        :param max_age:
        :return:
        """
        age_range = range(min_age, max_age + 1)
        df = pd.DataFrame(index=age_range, columns=age_range)

        for z in age_range:
            lower_bound = max(min_age, z - epsilon)
            upper_bound = min(max_age, z + epsilon)

            normalization = (upper_bound - lower_bound) + 1

            for y in age_range:
                df.at[y, z] = 1 / normalization if lower_bound <= y <= upper_bound else 0

        likelihood = df.stack()
        likelihood.index.names = [MP_INDEX_NAME, R_INDEX_NAME]
        likelihood.name = 'likelihood'

        return likelihood

    def _get_score_numerator(self):
        age_range = range(self.min_age, self.max_age + 1)

        # check if the ".cache" folder exists in nonGenCom, otherwise create it
        cache_path = os.path.join(os.path.dirname(__file__), '.cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # if there's a score_numerator_cache file, load it
        score_numerator_file_name = f'score_numerator_epsilon_{self.epsilon}.csv'
        if os.path.exists(os.path.join(cache_path, score_numerator_file_name)):
            score_numerator = pd.read_csv(os.path.join(cache_path, score_numerator_file_name), index_col=[0, 1])
            # limit the score_numerator to the age range
            filter_age_range = score_numerator.index.get_level_values(0).isin(age_range) & \
                               score_numerator.index.get_level_values(1).isin(age_range)
            score_numerator = score_numerator.loc[filter_age_range]
            return score_numerator

        # calculate the score_numerator
        score_numerator = pd.DataFrame(index=age_range,
                                       columns=age_range)

        for fc_age in age_range:
            for mp_age in age_range:
                res = sum(self.fc_likelihood.loc[fc_age] * self.mp_likelihood.loc[mp_age] * self.prior)

                score_numerator.iloc[fc_age, mp_age] = res

        score_numerator = score_numerator.stack()
        score_numerator.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]
        # save the score_numerator for future use
        score_numerator.to_csv(os.path.join(cache_path, score_numerator_file_name))

        return score_numerator

    def get_fc_score(self) -> Series:
        # TODO implement
        pass

    def get_mp_score(self, context_name: str, scenery_name: str) -> Series:
        # TODO implement
        pass
