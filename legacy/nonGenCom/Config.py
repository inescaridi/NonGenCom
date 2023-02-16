import os
import sys

import pandas as pd
from pandas import Series

from nonGenCom.Utils import convert_all_cells_to_float


class Config:
    def __init__(self, contexts_path="nonGenCom/default_inputs/biolsex_contexts.csv", sceneries_path="nonGenCom/default_inputs/biolsex_sceneries.csv"):
        # change working dir to "base" dir
        os.chdir(sys.path[0])
        if not os.path.isdir('nonGenCom'):  # may be already working on base dir
            os.chdir("..")

        cont_aux = pd.read_csv(contexts_path, skiprows=1, header=None).transpose()
        cont_aux[0] = cont_aux[0].str.upper()
        cont_columns = cont_aux.iloc[0]
        self.contexts = cont_aux.drop(index=0).set_axis(cont_columns, axis=1).set_index('MP')
        convert_all_cells_to_float(self.contexts)

        scen_aux = pd.read_csv(sceneries_path, skiprows=1, header=None).transpose()
        scen_aux[0] = scen_aux[0].str.upper()
        scen_aux[1] = scen_aux[1].str.upper()
        scen_columns = scen_aux.iloc[0]
        self.sceneries = scen_aux.drop(index=0).set_axis(scen_columns, axis=1).set_index(['FC', 'MP'])
        convert_all_cells_to_float(self.sceneries)
        # TODO add check for order of FC and MP

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


class TestConfig(Config):
    def __init__(self):
        super().__init__("tests/resources/context_examples.csv", "tests/resources/scenery_examples.csv")

        post_aux = pd.read_csv("tests/resources/posterior_examples.csv", skiprows=1, header=None).transpose()
        post_aux[0] = post_aux[0].str.upper()
        post_aux[1] = post_aux[1].str.upper()
        post_columns = post_aux.iloc[0]
        self.posteriors = post_aux.drop(index=0).set_axis(post_columns, axis=1).set_index(['FC', 'MP'])
        convert_all_cells_to_float(self.posteriors)

    def get_posterior(self, posterior_tag):
        return self.posteriors[posterior_tag]
