from typing import List

from pandas import Series

from nonGenCom.Variables.Variable import Variable


class BiologicalSex(Variable):
    SCORE_COLNAME = 'biolsex_score'

    def __init__(self, contexts_path="nonGenCom/default_inputs/biolsex_contexts.csv",
                 sceneries_path="nonGenCom/default_inputs/biolsex_sceneries.csv"):
        super().__init__(contexts_path, sceneries_path)

    def get_fc_posterior(self, context_name: str = None, scenery_name: str = None) -> Series:
        """
        Computes the posterior probability (the scores for FC-selection searches) for a given scenery and context.
        using the biological sex variable, given:
        # TODO update docstring
        :param context_name: representing P(MP= y) (context)
        :param scenery_name: representing P(FC= x | MP= y)  (scenery)
        :return: posterior: Series: representing P(FC=x |MP=y) * P(MP= x) / P(FC= y)
        """

        prior = self.get_context(context_name)
        likelihood = self.get_scenery(scenery_name)
        return self._calculate_bayes(prior, likelihood)

    def profiling(self, prior: Series, likelihood: Series, cos_pairs: List[str] = None, cow_pairs: List[str] = None,
                  ins_pairs: List[str] = None, inw_pairs: List[str] = None):
        """
        # TODO complete
        # TODO in general: add types to docstring
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
            cos_pairs = [('F', 'F'), ('M', 'M')]  # TODO move this "defaults" to config file

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
        # TODO move this to a configuration file
        return renames

    def get_fc_likelihood(self, scenery_name: str) -> Series:
        return self.get_scenery(scenery_name)
