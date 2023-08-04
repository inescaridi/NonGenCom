import pandas as pd
from pandas import Series, DataFrame

from nonGenCom.Utils import MP_INDEX_NAME, R_INDEX_NAME, FC_INDEX_NAME
from nonGenCom.Variables.AgeContinuous import AgeContinuous


class AgeMPRange(AgeContinuous):
    SCORE_COLNAME = 'age_v3_score'

    def __init__(self, context_name='Standard', epsilon=1, min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv",
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(context_name, min_age, max_age, contexts_path, sigmas_path)

        self.fc_likelihood = self.likelihood
        self.fc_likelihood.index.names = [FC_INDEX_NAME, R_INDEX_NAME]
        self.fc_likelihood.index = self.fc_likelihood.index.set_levels(self.fc_likelihood.index.levels[1].astype(int), level=1)
        self.fc_evidence = self.evidence

        self.mp_prior = self.prior.copy()
        self.mp_prior.index.names = [R_INDEX_NAME]
        self.mp_prior.index = self.mp_prior.index.astype(int)
        self.mp_likelihood = self.get_MP_likelihood(epsilon, min_age, max_age)
        self.mp_evidence = self._calculate_evidence(self.mp_prior, self.mp_likelihood)

        # Posteriors
        self.score_numerator = self._get_score_numerator()
        # posterior_numerator = self.fc_likelihood.mul(self.mp_likelihood.mul(self.mp_prior, level=1).groupby(level=1).sum(), level=1) # may be another way to do this
        self.fc_posteriors = {}
        self.mp_posteriors = {}

    def get_fc_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int):
        if any(pd.isna([fc_min_age, fc_max_age, mp_min_age, mp_max_age])):
            return None

        fc_key = (fc_min_age, fc_max_age)
        mp_key = (mp_min_age, mp_max_age)
        if fc_key in self.fc_posteriors:
            if mp_key in self.fc_posteriors[fc_key]:
                return self.fc_posteriors[fc_key][mp_key]

        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))
        fc_posterior_denominator = self.fc_evidence.loc[fc_age_range].sum()

        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))
        posterior_nominator = self.score_numerator.loc[fc_age_range, mp_age_range].sum()

        fc_posterior_value = posterior_nominator / fc_posterior_denominator
        self.fc_posteriors.setdefault(fc_key, {})[mp_key] = fc_posterior_value

        return fc_posterior_value

    def get_mp_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int):
        if any(pd.isna([fc_min_age, fc_max_age, mp_min_age, mp_max_age])):
            return None

        fc_key = (fc_min_age, fc_max_age)
        mp_key = (mp_min_age, mp_max_age)
        if mp_key in self.mp_posteriors:
            if fc_key in self.mp_posteriors[mp_key]:
                return self.mp_posteriors[mp_key][fc_key]

        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))

        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))
        mp_posterior_denominator = self.mp_evidence.loc[mp_age_range].sum()

        posterior_nominator = self.score_numerator.loc[fc_age_range, mp_age_range].sum()

        mp_posterior_value = posterior_nominator / mp_posterior_denominator
        self.mp_posteriors.setdefault(mp_key, {})[fc_key] = mp_posterior_value

        return mp_posterior_value

    # TODO move method to abstract class Variable if it makes sense
    def get_MP_likelihood(self, epsilon: int = 1, min_age: int = -1, max_age: int = 100) -> Series:
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
        mp_as_matrix = self.mp_likelihood.unstack()
        fc_as_matrix = self.fc_likelihood.unstack()
        score_numerator = mp_as_matrix.T.dot(fc_as_matrix.mul(self.mp_prior, axis=0)).T.stack()
        score_numerator.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]
        return score_numerator
