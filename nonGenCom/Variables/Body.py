from pandas import Series

from nonGenCom.CategoricalVariable import CategoricalVariable
from nonGenCom.Utils import get_md5_encoding


class Body(CategoricalVariable):
    def __init__(self, context_name: str, fc_scenery_name: str, mp_scenery_name: str, characteristic: str,
                 yes='YES', no='NO'):
        """

        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        :param characteristic:
        :param yes: the value that represents a positive answer
        :param no: the value that represents a negative answer
        """
        yes_no_categories = {yes, no}

        contexts_path = "nonGenCom/scenery_and_context_inputs/body_contexts.csv"
        fc_sceneries_path = "nonGenCom/scenery_and_context_inputs/body_fc_sceneries.csv"
        mp_sceneries_path = "nonGenCom/scenery_and_context_inputs/body_mp_sceneries.csv"

        self.characteristic = characteristic

        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, yes_no_categories, yes_no_categories, yes_no_categories)

    def _score_numerator_filename(self) -> str:
        fn = get_md5_encoding(self.context_name, self.fc_scenery_name, self.mp_scenery_name)
        return f"body_{fn}.csv"

    def score_colname_template(self) -> str:
        return "body_{}_score"

    def _reformat_prior(self, prior: Series):
        return prior

    def _get_fc_likelihood_for_combination(self, r_value: tuple, fc_value: tuple):
        return 0  # TODO fc likelihood calculation, right now we are only using pre-defined sceneries

    def _get_mp_likelihood_for_combination(self, r_value: tuple, mp_value: tuple):
        return 0  # TODO mp likelihood calculation, right now we are only using pre-defined sceneries
