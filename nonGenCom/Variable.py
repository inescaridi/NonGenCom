from typing import List

import pandas as pd
from pandas import Series

from nonGenCom.Utils import load_contexts, load_sceneries


class Variable:
    DECIMAL_PRECISION = 8

    def __init__(self, contexts_path, sceneries_path):
        self.contexts = load_contexts(contexts_path)
        self.sceneries = load_sceneries(sceneries_path) if sceneries_path is not None else pd.DataFrame()

        # TODO ask: should Variable receive context_name and scenery_name and save all information for next op?

    def get_posterior(self, context_name: str, scenery_name: str) -> Series:
        raise NotImplementedError

    def profiling(self, prior: Series, likelihood: Series, cos_pairs: List[str] = None, cow_pairs: List[str] = None,
                  ins_pairs: List[str] = None, inw_pairs: List[str] = None):
        raise NotImplementedError

    def get_context(self, context_name: str) -> Series:
        """
        Get context (aka prior)

        :param context_name: str:
        :return:
        """
        return self.contexts[context_name]

    def get_scenery(self, scenery_name: str) -> Series:
        """
        Get scenery (aka likelihood)

        :param scenery_name: str:
        :return:
        """
        return self.sceneries[scenery_name]

    def _calculate_bayes(self, prior: Series, likelihood: Series) -> Series:
        likelihood.fillna(0, inplace=True)  # TODO what should we do with NaN values?

        likelihood_x_prior = likelihood.multiply(prior, level=1)
        evidence = likelihood_x_prior.groupby('FC').sum()

        posterior = likelihood_x_prior.multiply(evidence ** -1, level=0)

        return posterior
