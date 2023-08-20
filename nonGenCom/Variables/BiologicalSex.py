from typing import List

import pandas as pd
from pandas import Series

from nonGenCom.CategoricalVariable import CategoricalVariable


class BiologicalSex(CategoricalVariable):
    def __init__(self, context_name: str, fc_scenery_name: str, mp_scenery_name: str):
        """

        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        """
        contexts_path = "nonGenCom/scenery_and_context_inputs/biolsex_contexts.csv"
        fc_sceneries_path = "nonGenCom/scenery_and_context_inputs/biolsex_fc_sceneries.csv"
        mp_sceneries_path = "nonGenCom/scenery_and_context_inputs/biolsex_mp_sceneries.csv"

        r_categories = self._get_categories_from_file(contexts_path)
        fc_categories = self._get_categories_from_file(fc_sceneries_path)
        mp_categories = self._get_categories_from_file(mp_sceneries_path)
        # TODO load from configuration file

        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name, r_categories, fc_categories, mp_categories)

    @staticmethod
    def _get_categories_from_file(filename) -> set:
        values = pd.read_csv(filename, header=None, index_col=0).iloc[0].values
        # TODO improve this or get information from configuration file
        return set(values)

    def score_colname_template(self) -> str:
        return 'biolsex_{}_score'

    def _get_fc_likelihood_for_combination(self, r_value: tuple, fc_value: tuple):
        return 0  # TODO fc likelihood calculation, right now we are only using pre-defined sceneries

    def _get_mp_likelihood_for_combination(self, r_value: tuple, mp_value: tuple):
        return 0  # TODO mp likelihood calculation, right now we are only using pre-defined sceneries

    def _reformat_prior(self, prior: Series):
        return prior

    @staticmethod
    def profiling(prior: Series, likelihood: Series, cos_pairs: List[str] = None, cow_pairs: List[str] = None,
                  ins_pairs: List[str] = None, inw_pairs: List[str] = None):
        """
        Computes the performance metrics Strong Consistency (CoS), Weak consistency (CoW), Strong Inconsistency (InS),
        and Weak Inconsistency (InW) for a given scenery and context based on the biological sex variable.

        :param likelihood: Series: representing P(FC= x | MP= y) (scenery)
        :param prior: Series: representing P(MP= y) (context)
        :param cos_pairs: List[str]: inputs to define Strong Consistency metric (CoS)
        :param cow_pairs: List[str]: inputs to define Weak Consistency metric (CoW)
        :param ins_pairs: List[str]: inputs to define Strong Inconsistency metric (InS)
        :param inw_pairs: List[str]: inputs to define Weak Inconsistency metric (InW)
        :return: metrics CoS, CoW, InS, InW
        """
        if cos_pairs is None:
            cos_pairs = [('F', 'F'), ('M', 'M')]

        if cow_pairs is None:
            cow_pairs = cos_pairs + [('PF', 'F'), ('PM', 'M'),
                                     ('F', 'O'), ('PF', 'O'), ('M', 'O'), ('PM', 'O'), ('I', 'O')]

        if ins_pairs is None:
            ins_pairs = [('M', 'F'), ('F', 'M')]

        if inw_pairs is None:
            inw_pairs = ins_pairs + [('PM', 'F'), ('PF', 'M')]

        cos = cow = ins = inw = 0

        for fc_value, mp_value in likelihood.index:
            value = likelihood[fc_value][mp_value] * prior[mp_value]
            pair = (fc_value, mp_value)

            if pair in cos_pairs:
                cos += value

            if pair in cow_pairs:
                cow += value

            if pair in ins_pairs:
                ins += value

            if pair in inw_pairs:
                inw += value

        return cos, cow, ins, inw

    @property
    def renames(self) -> dict[str, str]:
        renames = {
            'Indeterminate': 'i',
            'Probable Male': 'pm',
            'Probable Female': 'pf',
            'Male': 'm',
            'Female': 'f',
        }
        return renames
