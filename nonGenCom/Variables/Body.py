from pandas import Series

from nonGenCom.CategoricalVariable import CategoricalVariable


class Body(CategoricalVariable):
    def __init__(self, context_name: str, fc_scenery_name: str, mp_scenery_name: str, characteristic: str):
        """

        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        :param characteristic:
        """
        contexts_path = "nonGenCom/scenery_and_context_inputs/body_contexts.csv"
        fc_sceneries_path = "nonGenCom/scenery_and_context_inputs/body_fc_sceneries.csv"
        mp_sceneries_path = "nonGenCom/scenery_and_context_inputs/body_mp_sceneries.csv"

        self.characteristic = characteristic
        yes_no_categories = {'YES', 'NO'}

        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, yes_no_categories, yes_no_categories, yes_no_categories)

    def score_colname(self) -> str:
        return "body_score"

    def _reformat_prior(self, prior: Series):
        return prior

    def _get_fc_likelihood_for_combination(self, r_value: tuple, fc_value: tuple):
        return 0  # TODO fc likelihood calculation, right now we are only using pre-defined sceneries

    def _get_mp_likelihood_for_combination(self, r_value: tuple, mp_value: tuple):
        return 0  # TODO mp likelihood calculation, right now we are only using pre-defined sceneries
