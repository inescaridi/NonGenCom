import numpy as np
import pandas as pd

from nonGenCom.BiologicalSex import biolsex
from nonGenCom.Config import Config


class ScoreCalculator:
    def __init__(self):
        self.config = Config()

    def fc_score_biolsex(self, fc_db, mp_db, fc_element_id, context_name, scenery_name):
        self.context = self.config.getContext(context_name)  # prior
        self.scenery = self.config.getScenery(scenery_name)  # likelihood
        print(context_name, "\n", self.context)
        print(scenery_name, "\n", self.scenery)

        self.biolsex = biolsex(self.context, self.scenery)
        # TODO change biolsex to receive and return pandas dataframes as input (so we don't depend on order)

        fc_row = fc_db[fc_db.id == fc_element_id]

        mp_db['score'] = np.nan
        mp_db.apply(lambda mp_row: self._compare_fc_to_mp(mp_row, fc_row))

    def _compare_fc_to_mp(self, fc_row, mp_row):
        # TODO adapt the right part to match output from biolsex
        mp_row["score"] = self.biolsex[fc_row["biological_sex"]][mp_row["biological_sex"]]


if __name__ == '__main__':
    fc_db = pd.read_csv("default_inputs/fc_test.csv")
    mp_db = pd.read_csv("default_inputs/mp_test.csv")

    ScoreCalculator().fc_score_biolsex(fc_db, mp_db, "13", "Female bias", "Challenged")
