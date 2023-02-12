from typing import List

import pandas as pd
from pandas import Series


def biolsex(prior: Series, likelihood: Series) -> Series:
    """
    Computes the score for biological sex, given:
    # TODO update docstring
    :param likelihood: Series: representing P(FC= x | MP= y)
    :param prior: Series: representing P(MP= y)
    :return: posterior: Series: a Series representing P(FC=x |MP=y) * P(MP= x) / P(FC= y)
    """
    likelihood.fillna(0, inplace=True)  # TODO what should we do with NaN values?

    evidence = {}
    for fc_value, mp_value in likelihood.index:
        evidence.setdefault(fc_value, 0)
        evidence[fc_value] += likelihood[fc_value][mp_value] * prior[mp_value]

    posterior = []
    for fc_value, mp_value in likelihood.index:
        pos_value = round(likelihood[fc_value][mp_value] * prior[mp_value] / evidence[fc_value], 4)
        posterior.append({'FC': fc_value, 'MP': mp_value, 'posterior': pos_value})

    return pd.DataFrame(posterior).set_index(['FC', 'MP'])['posterior']


def profiling_biolsex(likelihood: Series, prior: Series,
                      cos_pairs: List[str] = None, cow_pairs: List[str] = None, ins_pairs: List[str] = None,
                      inw_pairs: List[str] = None):
    """
    # TODO complete
    # TODO in general: add types to docstring

    :param likelihood: Series: representing P(FC= x | MP= y)
    :param prior: Series: representing P(MP= y)
    :param cos_pairs: List[str]: the Strong Consistency metric (CoS) measures the probability of matching the sex of MP
    with that of FC.
    :param cow_pairs: List[str]: the Weak Consistency metric (CoW) measures the probability that the sex of FC or the
    potential sex of FC matches that of MP.
    :param ins_pairs: List[str]: the Strong Inconsistency metric (InS) measures the probability that the sex of MP
    does not coincide with the sex assigned to FC.
    :param inw_pairs: List[str]: the Weak Inconsistency metric (InW) measures the probability that the sex of MP
    does not coincide with the sex assigned to FC or the potential sex of FC.
    :return:
    """
    if cos_pairs is None:
        cos_pairs = [('F', 'F'), ('M', 'M')]  # TODO move this "defaults" to config file

    if cow_pairs is None:
        cow_pairs = cos_pairs + [('PF', 'F'), ('PM', 'M'),
                                 ('F', 'O'), ('PF', 'O'), ('M', 'O'), ('PM', 'O'), ('I', 'O')]

    if ins_pairs is None:
        ins_pairs = [('M', 'F'), ('F', 'M')]

    if inw_pairs is None:
        inw_pairs = ins_pairs + [('PM', 'F'), ('PF', 'M')]

    cos = cow = ins = inw = 0

    for fc_value, mp_value in likelihood.index:
        value = likelihood[fc_value][mp_value] * prior[mp_value]
        pair = (fc_value, mp_value)

        if pair in cos_pairs:
            cos += value

        if pair in cow_pairs:
            cow += value

        if pair in ins_pairs:
            ins += value

        if pair in inw_pairs:
            inw += value

    return cos, cow, ins, inw


