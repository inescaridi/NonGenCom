import numpy as np
import pandas as pd
from pandas import Series


def biolsex(prior: Series, likelihood: Series) -> Series:
    """
    Computes the score for biological sex, given:
    # TODO update docstring
    :param likelihood: an np.array of shape (n, m) representing P(FC= x | MP= y)
    :param prior: an np.array of shape (m, 1) representing P(MP= y)
    :return: posterior: an np.array of shape (n, m) representing P(FC=x |MP=y) * P(MP= x) / P(FC= y)
    """
    likelihood.fillna(0, inplace=True)  # TODO what should we do with NaN values?

    evidence = {}
    for fc_value, mp_value in likelihood.index:
        evidence.setdefault(fc_value, 0)
        evidence[fc_value] += likelihood[fc_value][mp_value] * prior[mp_value]

    posterior = []
    for fc_value, mp_value in likelihood.index:
        pos_value = likelihood[fc_value][mp_value] * prior[mp_value] / evidence[fc_value]
        posterior.append({'FC': fc_value, 'MP': mp_value, 'posterior': pos_value})

    return pd.DataFrame(posterior).set_index(['FC', 'MP'])['posterior']

