import pandas as pd

from .. import BaseScore



class BayesianScore(BaseScore):

    prior: pd.Series
    likelihood: pd.Series
    posterior: pd.DataFrame
    l_index: str
    t_index: str

    def __init__(self, prior: pd.Series, likelihood: pd.Series, l_index: str, t_index: str):
        self.prior = prior
        self.likelihood = likelihood
        self.l_index = l_index
        self.t_index = t_index
        self.posterior = self._get_posterior()
    
    def _get_posterior(self):
        
        evidence = {}
        for l_value, t_value in self.likelihood.index:
            evidence.setdefault(l_value, 0)
            evidence[l_value] += self.likelihood[l_value][t_value] * self.prior[t_value]

        posterior = []
        for l_value, t_value in self.likelihood.index:
            pos_value = round(self.likelihood[l_value][t_value] * self.prior[t_value] / evidence[l_value], 4)
            posterior.append({self.l_index: l_value, self.t_index: t_value, 'posterior': pos_value})

        return pd.DataFrame(posterior).set_index([self.l_index, self.t_index])['posterior']
    
    def __call__(self, item, **kw):
        # TODO get the attibute names from config
        res = None
        try:
            print(f"{item.req_biosex}, {item.src_biosex}")
            res = self.posterior[item.req_biosex][item.src_biosex]
        except KeyError:
            res = None

        return res