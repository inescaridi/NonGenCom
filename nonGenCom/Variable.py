from abc import abstractmethod, ABC
from functools import lru_cache

import pandas as pd
from pandas import Series, DataFrame

from nonGenCom.Utils import FC_INDEX_NAME, MP_INDEX_NAME, load_r_indexed_file, load_double_indexed_indexed_file, \
    R_INDEX_NAME


class Variable(ABC):
    DECIMAL_PRECISION = 8
    SCORE_COLNAME = 'BASE'

    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None):
        """
        :param mp_sceneries_path:
        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        :param contexts_path:
        :param fc_sceneries_path:
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

    def add_score_fc_by_merge(self, merged_dbs: DataFrame, fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        """
        # TODO complete docstring

        :param merged_dbs: databases already merged
        :param fc_value_colname: colname of value for variable in Fosensic Case Database
        :param mp_value_colname: colname of value for variable in Missing Person Database

        :return:
        """
        # TODO move all database "config" (column names mostly) to a class
        posterior = self.get_fc_score()
        print(f"Context: {self.context_name}")
        print(f"Scenery: {self.fc_scenery_name}")
        print("Posterior\n", posterior, "\n")

        merged_dbs = self._reindex(merged_dbs, fc_value_colname, mp_value_colname)

        # merge with posterior
        merged_dbs = merged_dbs.join(posterior.rename(self.SCORE_COLNAME)) \
            .reset_index(drop=True) \
            .sort_values(self.SCORE_COLNAME, ascending=False)

        return merged_dbs

    def add_score_mp_by_merge(self, merged_dbs: DataFrame, scenery_name: str,
                              fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        """
        # TODO complete docstring

        :param merged_dbs: databases already merged
        :param scenery_name:

        :param fc_value_colname: colname of value for variable in Fosensic Case Database
        :param mp_value_colname: colname of value for variable in Missing Person Database

        :return:
        """
        # TODO move all database "config" (column names mostly) to a class
        likelihood = self.get_fc_likelihood(scenery_name)
        print(f"Scenery: {scenery_name}")

        merged_dbs = self._reindex(merged_dbs, fc_value_colname, mp_value_colname)

        # merge with posterior
        merged_dbs = merged_dbs.join(likelihood.rename(self.SCORE_COLNAME)) \
            .reset_index(drop=True) \
            .sort_values(self.SCORE_COLNAME, ascending=False)

        return merged_dbs

    def _reindex(self, merged_dbs: DataFrame, fc_value_colname: str, mp_value_colname: str):
        # create new FC and MP columns, renaming if necessary
        if len(self.renames) > 0:
            merged_dbs['fc_index_aux'] = merged_dbs[fc_value_colname].astype(str).map(self.renames)
            merged_dbs['mp_index_aux'] = merged_dbs[mp_value_colname].astype(str).map(self.renames)
        else:
            merged_dbs['fc_index_aux'] = merged_dbs[fc_value_colname].astype(str)
            merged_dbs['mp_index_aux'] = merged_dbs[mp_value_colname].astype(str)

        # set the same index as posterior
        merged_dbs = merged_dbs.set_index(['fc_index_aux', 'mp_index_aux'])
        merged_dbs.index = merged_dbs.index.rename([FC_INDEX_NAME, MP_INDEX_NAME])
        return merged_dbs

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
            # TODO maybe raise Exception?
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

    @lru_cache(maxsize=128)
    def _get_score_numerator(self, fc_likelihood: Series, mp_likelihood: Series, prior: Series,
                             fc_values, mp_values) -> Series:
        score_numerator_dict = {}

        for fc_value in fc_values:
            for mp_value in mp_values:
                res = sum((fc_likelihood.loc[fc_value] * mp_likelihood.loc[mp_value] * prior).dropna())
                score_numerator_dict[(fc_value, mp_value)] = res

        score_numerator = pd.Series(score_numerator_dict)
        score_numerator.index.names = [FC_INDEX_NAME, MP_INDEX_NAME]
        # save the score_numerator for future use

        return score_numerator

    def get_prior(self, context_name: str) -> Series:
        prior = self.get_context(context_name)
        prior = self._reformat_prior(prior)
        return prior

    @classmethod
    def _calculate_evidence(cls, prior: Series, likelihood: Series) -> Series:
        likelihood_x_prior = cls._calculate_likelihood_x_prior(prior, likelihood)
        group_by = likelihood.index.levels[0].name
        evidence = likelihood_x_prior.groupby(group_by).sum()
        return evidence

    @classmethod
    def _calculate_likelihood_x_prior(cls, prior: Series, likelihood: Series) -> Series:
        if prior.index.dtype != likelihood.index.levels[1].dtype:
            print("WARNING: prior index type is not the same type as likelihood index. Converting prior index type to ")
            prior.index = prior.index.astype(likelihood.index.levels[1].dtype)

        likelihood_x_prior = likelihood.multiply(prior, level=1)
        return likelihood_x_prior

    @abstractmethod
    def get_fc_likelihood(self, scenery_name: str) -> Series:
        raise NotImplementedError

    @abstractmethod
    def get_mp_likelihood(self, scenery_name: str) -> Series:
        raise NotImplementedError

    @abstractmethod
    def _reformat_prior(self, prior: Series):
        """
        Returns the prior Series in the format expected by the subclass (but always a Series)
        :param prior: Series
        :return:
        """
        raise NotImplementedError()
