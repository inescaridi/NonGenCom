from typing import List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from nonGenCom.BiologicalSex import BiologicalSex


class ScoreCalculator:
    def __init__(self):
        self.biolsex_posterior = None

        self.biolsex_score_colname = 'score_biolsex'  # TODO move to BiologicalSex class

    def fc_score_biolsex(self, fc_db: DataFrame, mp_db: DataFrame, context_name: str, scenery_name: str,
                         fc_index_colname: str, fc_biolsex_colname: str, mp_biolsex_colname: str,
                         fc_elements_id: Optional[List] = None) -> DataFrame:
        """
        # TODO complete docstring

        :param fc_db: Forensic Case Database
        :param mp_db: Missing Person Database
        :param context_name: chosen context
        :param scenery_name: chosen scenery
        :param fc_index_colname: colname of Fosensic Case ID
        :param fc_biolsex_colname: colname of biological sex variable in Fosensic Case Database
        :param mp_biolsex_colname: colname of biological sex variable in Missing Person Database
        :param fc_elements_id: Optional[List]: if None is passed then uses full fc_db for for FC-selection comparison,
        otherwise filter by IDs included in fc_elements_id
        :return:
        """
        # TODO move all database "config" (column names mostly) to a class
        self.biolsex_posterior = BiologicalSex().get_posterior(context_name, scenery_name)
        print(f"Context: {context_name}")
        print(f"Scenery: {scenery_name}")
        print("Posterior\n", self.biolsex_posterior, "\n")

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
        merged = merged.join(self.biolsex_posterior).rename(columns={'posterior': self.biolsex_score_colname})\
            .reset_index(drop=True)\
            .sort_values(self.biolsex_score_colname, ascending=False)

        return merged

    @staticmethod
    def _get_biolsex_renames():
        renames = {
            'Indeterminate': 'i',
            'Probable Male': 'pm',
            'Probable Female': 'pf',
            'Male': 'm',
            'Female': 'f',
        }
        # TODO move this to a configuration file
        return renames
