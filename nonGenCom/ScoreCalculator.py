import numpy as np
import pandas as pd

from nonGenCom.BiologicalSex import biolsex
from nonGenCom.Config import Config


class ScoreCalculator:
    def __init__(self):
        self.config = Config()

    def fc_score_biolsex(self, fc_db, mp_db, fc_element_id, context_name, scenery_name,
                         fc_index_colname, fc_biolsex_colname, mp_biolsex_colname):
        # TODO move all database "config" (column names mostly) to a class
        self.context = self.config.get_context(context_name)  # prior
        self.scenery = self.config.get_scenery(scenery_name)  # likelihood
        print(context_name, "\n", self.context)
        print(scenery_name, "\n", self.scenery)

        self.biolsex = biolsex(self.context, self.scenery)
        print("Posterior", "\n", self.biolsex)

        fc_row = fc_db[fc_db[fc_index_colname] == fc_element_id]

        mp_db['score'] = mp_db.apply(lambda mp_row:
                                     self._compare_fc_to_mp(fc_row, mp_row, fc_biolsex_colname, mp_biolsex_colname),
                                     axis=1)

        return mp_db

    def _compare_fc_to_mp(self, fc_row, mp_row, fc_biolsex_colname, mp_biolsex_colname):
        # TODO adapt the right part to match output from biolsex
        fc_value = self._get_biolsex_index(fc_row[fc_biolsex_colname])
        mp_value = self._get_biolsex_index(mp_row[mp_biolsex_colname])

        return self.biolsex[fc_value][mp_value]

    @staticmethod
    def _get_biolsex_index(value):
        if isinstance(value, pd.Series):
            value = value[0]
        renames = {
            'Indeterminate': 'I',
            'Probable Male': 'PM',
            'Probable Female': 'PF',
            'Male': 'M',
            'Female': 'F',
        }
        # TODO move this to a configuration file
        return renames[value]
