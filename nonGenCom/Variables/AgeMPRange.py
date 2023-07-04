import pandas as pd
from pandas import Series, DataFrame

from nonGenCom.Utils import MP_INDEX_NAME, R_INDEX_NAME
from nonGenCom.Variables.AgeContinuous import AgeContinuous


class AgeMPRange(AgeContinuous):
    SCORE_COLNAME = 'age_v3_score'

    def __init__(self, context_name='Standard', epsilon=1, min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv",
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(context_name, min_age, max_age, contexts_path, sigmas_path)

        self.fc_likelihood = self.likelihood
        self.fc_evidence = self.evidence

        self.mp_prior = self.prior.copy()
        self.mp_prior.index.names = [R_INDEX_NAME]
        self.mp_likelihood = self.get_MP_likelihood(epsilon, min_age, max_age)
        self.mp_evidence = self._calculate_evidence(self.mp_prior, self.mp_likelihood)

    def get_posterior_for_case(self, fc_min_age: int, fc_max_age: int, mp_min_age: int, mp_max_age: int) -> float | None:
        # TODO refactor this method and/or class and parents
        if any(pd.isna([fc_min_age, fc_max_age, mp_min_age, mp_max_age])):
            return None

        fc_key = (fc_min_age, fc_max_age)
        mp_key = (mp_min_age, mp_max_age)
        if fc_key in self.posteriors:
            if mp_key in self.posteriors[fc_key]:
                return self.posteriors[fc_key][mp_key]

        fc_age_range = range(fc_min_age, fc_max_age + (1 if fc_min_age == fc_max_age else 0))
        mp_age_range = range(mp_min_age, mp_max_age + (1 if mp_min_age == mp_max_age else 0))

        l_filter = self.likelihood.index.get_level_values(0).isin(fc_age_range) & \
                 (self.likelihood.index.get_level_values(1).isin(mp_age_range))

        # TODO update calculation with v3

        sum_likelihoods_x_prior = sum(self.likelihood.loc[l_filter]) * self.prior[mp_age_range]

        posterior = round(sum_likelihoods_x_prior / self._get_evidence_for_range(fc_age_range), self.DECIMAL_PRECISION)

        self.posteriors.setdefault(fc_key, {})[mp_key] = posterior
        return posterior

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

