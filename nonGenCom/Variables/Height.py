import statistics as st

from numpy import int64
from pandas import Series

from nonGenCom.ContinuousVariable import ContinuousVariable
from nonGenCom.Utils import load_r_indexed_file, get_md5_encoding


class Height(ContinuousVariable):
    def __init__(self, context_name='Uniform', min_height: int = 50, max_height: int = 200, step: int = 1, epsilon=4):
        """

        :param context_name:
        :param min_height:
        :param max_height:
        :param step:
        :param epsilon:
        """
        contexts_path = "nonGenCom/scenery_and_context_inputs/height_contexts.csv"
        fc_sceneries_path = fc_scenery_name = None
        mp_sceneries_path = mp_scenery_name = None

        sigmas_path = "nonGenCom/scenery_and_context_inputs/height_sigma.csv"
        self.sigmas = load_r_indexed_file(sigmas_path)
        self.sigmas.index = self.sigmas.index.astype(int)

        self.epsilon = epsilon

        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, min_height, max_height, step)

    def _score_numerator_filename(self) -> str:
        fn = get_md5_encoding(self.context_name, self.min_value, self.max_value, self.step, self.epsilon)
        return f"height_{fn}.csv"

    def score_colname_template(self) -> str:
        return 'height_{}_score'

    def _reformat_prior(self, prior: Series | None):
        prior.index = prior.index.astype(int64)
        return prior

    def _get_fc_likelihood_for_combination(self, r_value: int, fc_value):
        sigma = float(self.sigmas.loc[r_value].iloc[0])
        normal_distribution = st.NormalDist(r_value, sigma)

        return normal_distribution.cdf(min(fc_value + 0.5, self.max_value)) - normal_distribution.cdf(max(fc_value - 0.5, self.min_value))

    def _get_mp_likelihood_for_combination(self, r_value, mp_value):
        lower_bound = max(self.min_value, r_value - self.epsilon)
        upper_bound = min(self.max_value, r_value + self.epsilon)

        return 1 if lower_bound <= mp_value <= upper_bound else 0

    def get_fc_score_for_range(self, fc_min_height: int, fc_max_height: int,
                               mp_min_height: int, mp_max_height: int) -> Series:
        """

        :param fc_min_height:
        :param fc_max_height:
        :param mp_min_height:
        :param mp_max_height:
        :return:
        """
        return self._calculate_fc_score_for_range(fc_min_height, fc_max_height, mp_min_height, mp_max_height)

    def get_mp_score_for_range(self, fc_min_height: int, fc_max_height: int,
                               mp_min_height: int, mp_max_height: int) -> Series:
        """

        :param fc_min_height:
        :param fc_max_height:
        :param mp_min_height:
        :param mp_max_height:
        :return:
        """
        return self._calculate_mp_score_for_range(fc_min_height, fc_max_height, mp_min_height, mp_max_height)
