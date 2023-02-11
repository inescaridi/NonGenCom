from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from nonGenCom.BiologicalSex import biolsex
from nonGenCom.Config import Config


class ScoreCalculator:
    def __init__(self):
        self.config = Config()
        self.biolsex = None

        self.biolsex_score_colname = 'score_biolsex'

    def fc_score_biolsex(self, fc_db: DataFrame, mp_db: DataFrame, context_name: str, scenery_name: str,
                         fc_index_colname: str, fc_biolsex_colname: str, mp_biolsex_colname: str,
                         fc_elements_id: Optional[List] = None) -> DataFrame:
        """
        # TODO complete docstring

        :param fc_db:
        :param mp_db:
        :param context_name:
        :param scenery_name:
        :param fc_index_colname:
        :param fc_biolsex_colname:
        :param mp_biolsex_colname:
        :param fc_elements_id: Optional[List]: if None is passed then uses full fc_db for comparison,
        otherwise filter by IDs included in fc_elements_id
        :return:
        """
        # TODO move all database "config" (column names mostly) to a class
        context = self.config.get_context(context_name)  # prior
        scenery = self.config.get_scenery(scenery_name)  # likelihood
        print(f"Context: {context_name}\n", context, "\n")
        print(f"Scenery: {scenery_name}\n", scenery, "\n")

        self.biolsex = biolsex(context, scenery)
        print("Posterior\n", self.biolsex, "\n")

        if fc_elements_id is None:
            fc_rows = fc_db.copy()
        else:
            fc_rows: DataFrame = fc_db[fc_db[fc_index_colname].isin(fc_elements_id)]

        merged = fc_rows.merge(mp_db, how='cross', suffixes=('_FC', '_MP'))

        # BIOLSEX SCORE CALCULATION
        if fc_biolsex_colname == mp_biolsex_colname:  # there can be a collision on merge
            fc_biolsex_colname += '_FC'
            mp_biolsex_colname += '_MP'

        # create new FC and MP biological sex columns according to the renaming dict returned by _get_biolsex_renames
        merged['fc_biolsex_index_aux'] = merged[fc_biolsex_colname].map(self._get_biolsex_renames())
        merged['mp_biolsex_index_aux'] = merged[mp_biolsex_colname].map(self._get_biolsex_renames())

        # set the same index as biolsex
        merged = merged.set_index(['fc_biolsex_index_aux', 'mp_biolsex_index_aux'])
        merged.index = merged.index.rename(['FC', 'MP'])

        # merge with biolsex
        merged = merged.join(self.biolsex).rename(columns={'posterior': self.biolsex_score_colname})\
            .reset_index(drop=True)\
            .sort_values(self.biolsex_score_colname, ascending=False)

        return merged

    @staticmethod
    def _get_biolsex_renames():
        renames = {
            'Indeterminate': 'I',
            'Probable Male': 'PM',
            'Probable Female': 'PF',
            'Male': 'M',
            'Female': 'F',
        }
        # TODO move this to a configuration file
        return renames
