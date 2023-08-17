from abc import abstractmethod, ABC
from functools import lru_cache

import pandas as pd
from pandas import Series

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME, load_r_indexed_file, load_double_indexed_indexed_file, \
    R_INDEX_NAME


class Variable(ABC):
    DECIMAL_PRECISION = 8
    SCORE_COLNAME = 'BASE'

    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None):
        """

        :param contexts_path:
        :param fc_sceneries_path:
        :param mp_sceneries_path:
        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        """
        self.context_name = context_name
        self.fc_scenery_name = fc_scenery_name
        self.mp_scenery_name = mp_scenery_name

        self.contexts = pd.DataFrame()
        self.fc_sceneries = pd.DataFrame()
        self.mp_sceneries = pd.DataFrame()

        if contexts_path is not None:
            try:
                self.contexts = load_r_indexed_file(contexts_path)
            except FileNotFoundError:
                print(f"Contexts file not found: {contexts_path}")

        if fc_sceneries_path is not None:
            try:
                self.fc_sceneries = load_double_indexed_indexed_file(fc_sceneries_path,
                                                                     'FC', FC_INDEX_NAME,
                                                                     'R', R_INDEX_NAME)
            except FileNotFoundError:
                print(f"FC Sceneries file not found: {fc_sceneries_path}")

        if mp_sceneries_path is not None:
            try:
                self.mp_sceneries = load_double_indexed_indexed_file(mp_sceneries_path,
                                                                     'MP', MP_INDEX_NAME,
                                                                     'R', R_INDEX_NAME)
            except FileNotFoundError:
                print(f"MP Sceneries file not found: {mp_sceneries_path}")

        super().__init__()

    @property
    def renames(self) -> dict[str, str]:
        return {}

    def get_context(self, context_name: str) -> Series | None:
        """
        Get context (aka prior)

        :param context_name: str:
        :return:
        """
        if context_name in self.contexts:
            return self.contexts[context_name]
        else:
            return None

    def get_fc_scenery(self, scenery_name: str) -> Series | None:
        """
        Get scenery (aka likelihood)

        :param scenery_name: str:
        :return:
        """
        if scenery_name is not None and scenery_name in self.fc_sceneries:
            return self.fc_sceneries[scenery_name]
        else:
            return None

    def get_mp_scenery(self, scenery_name: str) -> Series | None:
        """
        Get scenery (aka likelihood)

        :param scenery_name: str:
        :return:
        """
        if scenery_name is not None and scenery_name in self.mp_sceneries:
            return self.mp_sceneries[scenery_name]
        else:
            return None

    def _get_score_numerator(self, fc_likelihood: Series, mp_likelihood: Series, prior: Series,
                             fc_values, mp_values) -> Series:
        """

        :param fc_likelihood:
        :param mp_likelihood:
        :param prior:
        :param fc_values:
        :param mp_values:
        :return:
        """
        score_numerator_dict = {}

        for fc_value in fc_values:
            for mp_value in mp_values:
                res = sum((fc_likelihood.loc[fc_value] * mp_likelihood.loc[mp_value] * prior).dropna())
                score_numerator_dict[(fc_value, mp_value)] = res

        score_numerator = pd.Series(score_numerator_dict)
        score_numerator.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]
        # save the score_numerator for future use

        return score_numerator

    def get_prior(self, context_name: str = None) -> Series:
        """

        :param context_name:
        :return:
        """
        prior = self.get_context(context_name)
        prior = self._reformat_prior(prior)
        if prior is None:
            raise ValueError(f"Prior is not defined for context")
        return prior

    def _calculate_fc_likelihood(self, scenery_name, fc_values, r_values) -> Series:
        """

        :param scenery_name:
        :param fc_values:
        :param r_values:
        :return:
        """
        scenery = self.get_fc_scenery(scenery_name)
        if scenery is not None:
            return scenery

        likelihood_list = []
        for fc_category in fc_values:
            for r_category in r_values:
                likelihood_value = self._get_fc_likelihood_for_combination(r_category, fc_category)
                likelihood_list.append({FC_INDEX_NAME: fc_category, R_INDEX_NAME: r_category,
                                        'likelihood': likelihood_value})

        likelihood = pd.DataFrame(likelihood_list).set_index([FC_INDEX_NAME, R_INDEX_NAME])['likelihood']
        return likelihood

    def _calculate_mp_likelihood(self, scenery_name, mp_values, r_values):
        """

        :param scenery_name:
        :param mp_values:
        :param r_values:
        :return:
        """
        scenery = self.get_mp_scenery(scenery_name)
        if scenery is not None:
            return scenery

        likelihood_list = []
        for mp_category in mp_values:
            for r_category in r_values:
                likelihood_value = self._get_mp_likelihood_for_combination(r_category, mp_category)
                likelihood_list.append({MP_INDEX_NAME: mp_category, R_INDEX_NAME: r_category,
                                        'likelihood': likelihood_value})

        likelihood = pd.DataFrame(likelihood_list).set_index([MP_INDEX_NAME, R_INDEX_NAME])['likelihood']
        return likelihood

    @classmethod
    def _calculate_evidence(cls, prior: Series, likelihood: Series) -> Series:
        """

        :param prior:
        :param likelihood:
        :return:
        """
        likelihood_x_prior = cls._calculate_likelihood_x_prior(prior, likelihood)
        group_by = likelihood.index.levels[0].name
        evidence = likelihood_x_prior.groupby(group_by).sum()
        return evidence

    @classmethod
    def _calculate_likelihood_x_prior(cls, prior: Series, likelihood: Series) -> Series:
        """

        :param prior:
        :param likelihood:
        :return:
        """
        if prior.index.dtype != likelihood.index.levels[1].dtype:
            print("WARNING: prior index type is not the same type as likelihood index. Converting prior index type to ")
            prior.index = prior.index.astype(likelihood.index.levels[1].dtype)

        likelihood_x_prior = likelihood.multiply(prior, level=1)
        return likelihood_x_prior

    @property
    @abstractmethod
    def score_colname(self) -> str:
        """
        Name of the column for this variable score in the final dataframe
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_fc_likelihood(self, scenery_name: str = None) -> Series:
        """

        :param scenery_name:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_mp_likelihood(self, scenery_name: str = None) -> Series:
        """

        :param scenery_name:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _reformat_prior(self, prior: Series | None):
        """
        Returns the prior Series in the format expected by the subclass (but always a Series)
        :param prior: Series
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_fc_likelihood_for_combination(self, r_category, fc_category):
        """

        :param r_category:
        :param fc_category:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_mp_likelihood_for_combination(self, r_category, mp_category):
        """

        :param r_category:
        :param mp_category:
        :return:
        """
        raise NotImplementedError()
