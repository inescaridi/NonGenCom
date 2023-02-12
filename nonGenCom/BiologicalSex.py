from typing import List

import pandas as pd
from pandas import Series


def biolsex(prior: Series, likelihood: Series) -> Series:
    """
    Computes the the posterior probability (the scores for FC-selection searches) for a given scenery and context.
    using the biological sex variable, given:
    # TODO update docstring
    :param likelihood: Series: representing P(FC= x | MP= y)  (scenery)
    :param prior: Series: representing P(MP= y) (context)
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
    Computes the performance metrics Strong Consistency (CoS), Weak consistency (CoW), Strong Inconsistency (InS),
    and Weak Inconsistency (InW) for a given scenery and context based on the biological sex variable.
   
    :param likelihood: Series: representing P(FC= x | MP= y) (scenery)
    :param prior: Series: representing P(MP= y) (context)
    :param cos_pairs: List[str]: inputs to define Strong Consistency metric (CoS)
    :param cow_pairs: List[str]: inputs to define Weak Consistency metric (CoW)
    :param ins_pairs: List[str]: inputs to define Strong Inconsistency metric (InS)
    :param inw_pairs: List[str]: inputs to define Weak Inconsistency metric (InW)
    :return: metrics CoS, CoW, InS, InW
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


