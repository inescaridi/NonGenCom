import os
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

        :param contexts_path: path of the context input file 
        :param fc_sceneries_path: path of the FC scenerie input file
        :param mp_sceneries_path: path of the MP scenerie input file
        :param context_name:  name of the context input file 
        :param fc_scenery_name: name of the FC scenerie input file
        :param mp_scenery_name: name of the MP scenerie input file
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

    def get_context(self, context_name: str) -> Series | None:
        """
        Get context (aka prior of R variable)

        :param context_name: str: name of the context input file 
        :return:
        """
        if context_name in self.contexts:
            return self.contexts[context_name]
        else:
            return None

    def get_fc_scenery(self, scenery_name: str) -> Series | None:
        """
        Get scenery (aka FC-likelihood)
        :param scenery_name: str: name of the FC scenerie input file
        :return:
        """
        if scenery_name is not None and scenery_name in self.fc_sceneries:
            return self.fc_sceneries[scenery_name]
        else:
            return None

    def get_mp_scenery(self, scenery_name: str) -> Series | None:
        """
        Get scenery (aka MP-likelihood)
        :param scenery_name: str: name of the MP scenerie input file
        :return:
        """
        if scenery_name is not None and scenery_name in self.mp_sceneries:
            return self.mp_sceneries[scenery_name]
        else:
            return None

    def _get_score_numerator(self, fc_likelihood: Series, mp_likelihood: Series, prior: Series,
                             fc_values, mp_values) -> Series:
        """
        Calculate score_numerator
        
        :param fc_likelihood: FC-Likelihood P(FC=i | R=k)
        :param mp_likelihood: MP-Likelihood P(MP=j | R=k)
        :param prior: Prioris P(R=k)
        :param fc_values: possible values of FC
        :param mp_values: possible values of MP
        :return:
        """
        cache_path = os.path.join(os.path.dirname(__file__), 'Variables/.cache')
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        # if there's a score_numerator_cache file, load it
        score_numerator_file_name = self._score_numerator_filename()
        if os.path.exists(os.path.join(cache_path, score_numerator_file_name)):
            score_numerator = pd.read_csv(os.path.join(cache_path, score_numerator_file_name), index_col=[0, 1])['values']
            return score_numerator

        # calculate the score_numerator
        score_numerator_dict = {}

        for fc_value in fc_values:
            for mp_value in mp_values:
                res = sum((fc_likelihood.loc[fc_value] * mp_likelihood.loc[mp_value] * prior).dropna())
                score_numerator_dict[(fc_value, mp_value)] = res

        score_numerator = pd.Series(score_numerator_dict)
        score_numerator.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]

        # save the score_numerator for future use
        score_numerator.to_csv(os.path.join(cache_path, score_numerator_file_name), header=['values'])

        return score_numerator

    def get_prior(self, context_name: str = None) -> Series:
        """
        Get Prior (aka prior of R variable)

        :param context_name: name of the context input file 
        :return:
        """
        prior = self.get_context(context_name)
        prior = self._reformat_prior(prior)
        if prior is None:
            raise ValueError(f"Prior is not defined for context")
        return prior

    def _calculate_fc_likelihood(self, scenery_name, fc_values, r_values) -> Series:
        """
        Calculate FC-Likelihood  P(FC=i | R=k)
        :param scenery_name: name of the FC scenerie input file
        :param fc_values: possible values of FC
        :param r_values: possible values of R
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
        # normalize by R_INDEX_NAME sum
        likelihood = likelihood.groupby(R_INDEX_NAME).transform(lambda x: x / x.sum())
        return likelihood

    def _calculate_mp_likelihood(self, scenery_name, mp_values, r_values):
        """
        Calculate MP-Likelihood  P(MP=j | R=k)
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
        # normalize by R_INDEX_NAME sum
        likelihood = likelihood.groupby(R_INDEX_NAME).transform(lambda x: x / x.sum())
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
            print(f"WARNING: prior index type ({prior.index.dtype}) is not the same type as likelihood index "
                  f"({likelihood.index.levels[1].dtype}). Converting")
            prior.index = prior.index.astype(likelihood.index.levels[1].dtype)

        likelihood_x_prior = likelihood.multiply(prior, level=1)
        return likelihood_x_prior

    @abstractmethod
    def score_colname_template(self) -> str:
        """
        Name of the column for this variable score in the final dataframe
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _score_numerator_filename(self) -> str:
        """
        Name of the file for this variable score numerator, should depend on the interfering parameters
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_fc_likelihood(self, scenery_name: str = None) -> Series:
        """
        Get FC-Likelihood  P(FC=i | R=k)
        :param scenery_name: name of the FC scenerie input file
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_mp_likelihood(self, scenery_name: str = None) -> Series:
        """
        Get MP-Likelihood  P(MP=j | R=k)
        :param scenery_name: name of the MP scenerie input file
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
    def _get_fc_likelihood_for_combination(self, r_value, fc_value):
        """

        :param r_value:
        :param fc_value:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _get_mp_likelihood_for_combination(self, r_value, mp_value):
        """

        :param r_value:
        :param mp_value:
        :return:
        """
        raise NotImplementedError()
