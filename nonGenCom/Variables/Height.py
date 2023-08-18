import statistics as st
from pandas import Series

from nonGenCom.ContinuousVariable import ContinuousVariable
from nonGenCom.Utils import load_mp_indexed_file


class Height(ContinuousVariable):
    def __init__(self, context_name='Standard', min_height: int = 50, max_height: int = 200, step: int = 1, epsilon=4):
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
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, min_height, max_height, step)

        sigmas_path = "nonGenCom/scenery_and_context_inputs/height_sigma.csv"
        self.sigmas = load_mp_indexed_file(sigmas_path)
        self.sigmas.index = self.sigmas.index.astype(int)

        self.epsilon = epsilon

    @property
    def score_colname(self) -> str:
        return 'height_score'

    def _reformat_prior(self, prior: Series | None):
        return prior

    def _get_fc_likelihood_for_combination(self, r_value: int, fc_value):
        sigma = float(self.sigmas.loc[r_value].iloc[0])
        normal_distribution = st.NormalDist(r_value, sigma)

        upper = normal_distribution.cdf(min(fc_value + 0.5, self.max_value)) - normal_distribution.cdf(max(fc_value - 0.5, self.min_value))
        lower = normal_distribution.cdf(self.max_value) - normal_distribution.cdf(self.min_value)
        return upper / lower

    def _get_mp_likelihood_for_combination(self, r_value, mp_value):
        pass

    def get_fc_score_for_range(self, fc_min_value, fc_max_value, mp_min_value, mp_max_value) -> Series:
        pass

    def get_mp_score_for_range(self, fc_min_value, fc_max_value, mp_min_value, mp_max_value) -> Series:
        pass
