from typing import List

import pandas as pd
from pandas import Series, DataFrame

from nonGenCom.Utils import load_contexts, load_sceneries, FC_INDEX_NAME, MP_INDEX_NAME


class Variable:
    DECIMAL_PRECISION = 8
    SCORE_COLNAME = 'BASE'

    def __init__(self, contexts_path: str, sceneries_path: str):
        """
        :param contexts_path:
        :param sceneries_path:
        """
        self.contexts = pd.DataFrame()
        self.sceneries = pd.DataFrame()

        if contexts_path is not None:
            try:
                self.contexts = load_contexts(contexts_path)
            except FileNotFoundError:
                print(f"Contexts file not found: {contexts_path}")

        if sceneries_path is not None:
            try:
                self.sceneries = load_sceneries(sceneries_path)
            except FileNotFoundError:
                print(f"Sceneries file not found: {sceneries_path}")

    def add_score_fc_by_merge(self, merged_dbs: DataFrame, context_name: str, scenery_name: str,
                              fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        """
        # TODO complete docstring

        :param merged_dbs: databases already merged
        :param context_name:
        :param scenery_name:

        :param fc_value_colname: colname of value for variable in Fosensic Case Database
        :param mp_value_colname: colname of value for variable in Missing Person Database

        :return:
        """
        # TODO move all database "config" (column names mostly) to a class
        posterior = self.get_posterior(context_name, scenery_name)
        print(f"Context: {context_name}")
        print(f"Scenery: {scenery_name}")
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
        likelihood = self.get_scenery(scenery_name)
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
        evidence = likelihood_x_prior.groupby(FC_INDEX_NAME).sum()

        posterior = likelihood_x_prior.multiply(evidence ** -1, level=0)

        return posterior
