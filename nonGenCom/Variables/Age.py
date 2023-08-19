import statistics as st

from pandas import Series

from nonGenCom.ContinuousVariable import ContinuousVariable
from nonGenCom.Utils import load_mp_indexed_file, load_r_indexed_file


class Age(ContinuousVariable):
    def __init__(self, context_name='Standard', min_age: int = -1, max_age: int = 100, step: int = 1, epsilon=1):
        """

        :param context_name:
        :param min_age:
        :param max_age:
        :param step:
        :param epsilon:
        """
        contexts_path = "nonGenCom/scenery_and_context_inputs/age_contexts.csv"
        fc_sceneries_path = fc_scenery_name = None
        mp_sceneries_path = mp_scenery_name = None
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, min_age, max_age, step)

        sigmas_path = "nonGenCom/scenery_and_context_inputs/age_sigma.csv"
        self.sigmas = load_r_indexed_file(sigmas_path)
        self.sigmas.index = self.sigmas.index.astype(int)

        self.epsilon = epsilon

    def score_colname(self) -> str:
        return 'age_score'

    def _get_fc_likelihood_for_combination(self, r_value: int, fc_value: int):
        sigma = float(self.sigmas.loc[r_value].iloc[0])
        normal_distribution = st.NormalDist(r_value, sigma)

        return normal_distribution.cdf(min(fc_value + 0.5, self.max_value)) - normal_distribution.cdf(max(fc_value - 0.5, self.min_value))

    def _get_mp_likelihood_for_combination(self, r_value: int, mp_value: int):
        lower_bound = max(self.min_value, r_value - self.epsilon)
        upper_bound = min(self.max_value, r_value + self.epsilon)

        return 1 if lower_bound <= mp_value <= upper_bound else 0

    def get_fc_score_for_range(self, fc_min_age: int, fc_max_age: int,
                               mp_min_age: int, mp_max_age: int) -> Series:
        """

        :param fc_min_age:
        :param fc_max_age:
        :param mp_min_age:
        :param mp_max_age:
        :return:
        """
        return self._calculate_fc_score_for_range(fc_min_age, fc_max_age, mp_min_age, mp_max_age)

    def get_mp_score_for_range(self, fc_min_age: int, fc_max_age: int,
                               mp_min_age: int, mp_max_age: int) -> Series:
        """

        :param fc_min_age:
        :param fc_max_age:
        :param mp_min_age:
        :param mp_max_age:
        :return:
        """
        return self._calculate_mp_score_for_range(fc_min_age, fc_max_age, mp_min_age, mp_max_age)

    def _reformat_prior(self, prior: Series):
        return prior
