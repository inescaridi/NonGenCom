import pandas as pd


class Config:
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

    def getContext(self, context_name):  # prior
        return self.contexts[context_name]

    def getScenery(self, scenery_name):  # likelihood
        return self.sceneries[scenery_name]

    def getPosterior(self, posterior_tag):
        return self.posteriors[posterior_tag]

class TestConfig(Config):
    def __init__(self):
        cont_aux = pd.read_csv("../tests/resources/context_examples.csv", skiprows=1, header=None).transpose() # TODO change working directory to "base"
        cont_aux[0] = cont_aux[0].str.upper()
        cont_columns = cont_aux.iloc[0]
        self.contexts = cont_aux.drop(index=0).set_axis(cont_columns, axis=1).set_index('MP')

        scen_aux = pd.read_csv("../tests/resources/scenery_examples.csv", skiprows=1, header=None).transpose()
        scen_aux[0] = scen_aux[0].str.upper()
        scen_aux[1] = scen_aux[1].str.upper()
        scen_columns = scen_aux.iloc[0]
        self.sceneries = scen_aux.drop(index=0).set_axis(scen_columns, axis=1).set_index(['FC', 'MP'])
        # TODO add check for order of FC and MP

        post_aux = pd.read_csv("../tests/resources/posterior_examples.csv", skiprows=1, header=None).transpose()
        post_aux[0] = post_aux[0].str.upper()
        post_aux[1] = post_aux[1].str.upper()
        post_columns = post_aux.iloc[0]
        self.posteriors = post_aux.drop(index=0).set_axis(post_columns, axis=1).set_index(['FC', 'MP'])
