from abc import ABC

import numpy as np
from pandas import Series, DataFrame

from nonGenCom.Utils import MP_INDEX_NAME, FC_INDEX_NAME
from nonGenCom.Variable import Variable


class CategoricalVariable(Variable, ABC):
    def __init__(self, contexts_path: str | None, fc_sceneries_path: str | None, mp_sceneries_path: str | None,
                 context_name: str | None, fc_scenery_name: str | None, mp_scenery_name: str | None,
                 r_categories: set, fc_categories: set, mp_categories: set):
        """

        :param contexts_path:
        :param fc_sceneries_path:
        :param mp_sceneries_path:
        :param context_name:
        :param fc_scenery_name:
        :param mp_scenery_name:
        :param r_categories:
        :param fc_categories:
        :param mp_categories:
        """
        super().__init__(contexts_path, fc_sceneries_path, mp_sceneries_path, context_name, fc_scenery_name,
                         mp_scenery_name)
        self.r_categories = r_categories
        self.fc_categories = fc_categories
        self.mp_categories = mp_categories

        self.prior = self.get_prior(context_name)
        self.fc_likelihood = self.get_fc_likelihood(fc_scenery_name)
        self.mp_likelihood = self.get_mp_likelihood(mp_scenery_name)

        self.score_numerator = self._get_score_numerator(self.fc_likelihood,
                                                         self.mp_likelihood,
                                                         self.prior,
                                                         self.fc_categories,
                                                         self.mp_categories)
        self.fc_score = self.get_fc_score()
        self.mp_score = self.get_mp_score()

    def get_fc_likelihood(self, scenery_name: str = None) -> Series:
        return self._calculate_fc_likelihood(scenery_name, self.fc_categories, self.r_categories)

    def get_mp_likelihood(self, scenery_name: str = None) -> Series:
        return self._calculate_mp_likelihood(scenery_name, self.mp_categories, self.r_categories)

    def get_fc_score(self) -> Series:
        """

        :return:
        """
        evidence = self._calculate_evidence(self.prior, self.fc_likelihood)
        evidence.replace(0, np.NAN, inplace=True)  # avoid division by zero
        return self.score_numerator.divide(evidence, level=FC_INDEX_NAME)

    def get_mp_score(self) -> Series:
        """

        :return:
        """
        evidence = self._calculate_evidence(self.prior, self.mp_likelihood)
        evidence.replace(0, np.NAN, inplace=True)  # avoid division by zero
        return self.score_numerator.divide(evidence, level=MP_INDEX_NAME)

    def get_fc_score_for_combination(self, fc_category: str, mp_category: str) -> float:
        """

        :param fc_category:
        :param mp_category:
        :return:
        """
        return self.fc_score.loc[(fc_category, mp_category)]

    def get_mp_score_for_combination(self, fc_category: str, mp_category: str) -> float:
        """

        :param fc_category:
        :param mp_category:
        :return:
        """
        return self.mp_score.loc[(fc_category, mp_category)]

    def add_fc_score(self, merged_dbs: DataFrame, fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        """

        :param merged_dbs: databases already merged
        :param fc_value_colname: colname of value for variable in Fosensic Case Database
        :param mp_value_colname: colname of value for variable in Missing Person Database

        :return:
        """
        merged_dbs = self._reindex(merged_dbs, fc_value_colname, mp_value_colname)
        score_colname = self.score_colname_template().format('fc')

        # merge with posterior
        merged_dbs = merged_dbs.join(self.fc_score.rename(score_colname)) \
            .reset_index(drop=True) \
            .sort_values(score_colname, ascending=False)

        return merged_dbs

    def add_mp_score(self, merged_dbs: DataFrame, fc_value_colname: str, mp_value_colname: str) -> DataFrame:
        """

        :param merged_dbs: databases already merged
        :param fc_value_colname: colname of value for variable in Fosensic Case Database
        :param mp_value_colname: colname of value for variable in Missing Person Database

        :return:
        """
        merged_dbs = self._reindex(merged_dbs, fc_value_colname, mp_value_colname)
        score_colname = self.score_colname_template().format('mp')

        # merge with posterior
        merged_dbs = merged_dbs.join(self.mp_score.rename(score_colname)) \
            .reset_index(drop=True) \
            .sort_values(score_colname, ascending=False)

        return merged_dbs

    @property
    def renames(self) -> dict[str, str]:
        return {}

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

