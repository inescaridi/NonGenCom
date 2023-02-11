import numpy as np
import statistics as st
import math

minAge = -1
maxAge = 100


def age_conditional_probability(category, missing_person_age):
    """
    The method computes de conditional probability of a chosen range of ages given it was a particular age
    or more formally:   P(FC = category | MP = missing_person_age)
    :param category: corresponds to a set of two values, representing a gap of ages (a,b), a < b
    :param missing_person_age: number between min and max age, representing the actual age of the remains
    :return:
    """
    mu, sigma = missing_person_age, math.sqrt(obtain_sigma(missing_person_age))
    normal_distribution = st.NormalDist(mu, sigma)
    upper = normal_distribution.cdf(category[1]) - normal_distribution.cdf(category[0])
    lower = normal_distribution.cdf(maxAge) - normal_distribution.cdf(minAge)
    return upper/lower



def obtain_sigma(age):
    """
    according to age, obtains value of sigma (or variance for normal distribution)
    :param age:
    :return:
    """
    if minAge <= age or age <= 1:
        return 0.5
    elif 1 < age or age <= 15:
        return 2
    else:
        return 5
