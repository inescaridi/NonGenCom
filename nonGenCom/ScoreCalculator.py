import numpy as np
import pandas as pd

from nonGenCom.BiologicalSex import biolsex


class ScoreCalculator:
    def __init__(self):
        cont_aux = pd.read_csv("default_inputs/contexts.csv", skiprows=1, header=None).transpose()
        cont_aux[0] = cont_aux[0].str.upper()
        cont_columns = cont_aux.iloc[0]
        self.contexts = cont_aux.drop(index=0).set_axis(cont_columns, axis=1).set_index('MP')

        scen_aux = pd.read_csv("default_inputs/sceneries.csv", skiprows=1, header=None).transpose()
        scen_aux[0] = scen_aux[0].str.upper()
        scen_aux[1] = scen_aux[1].str.upper()
        scen_columns = scen_aux.iloc[0]
        self.sceneries = scen_aux.drop(index=0).set_axis(scen_columns, axis=1).set_index(['FC', 'MP'])
        # TODO add check for order of FC and MP

        self.biolsex = None

    def fc_score_biolsex(self, fc_db, mp_db, fc_element_id, context_name, scenery_name):
        self.context = self.contexts[context_name]  # prior
        self.scenery = self.sceneries[scenery_name]  # likelihood
        print(context_name, "\n", self.context)
        print(scenery_name, "\n", self.scenery)

        self.biolsex = biolsex(self.scenery, self.context)
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
