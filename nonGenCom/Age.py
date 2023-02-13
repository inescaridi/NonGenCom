import statistics as st
import math


def age_conditional_probability(min_age, max_age, category, missing_person_age):
    """
    The method computes de conditional probability of a chosen range of ages given it was a particular age
    or more formally:   P(FC = category | MP = missing_person_age)
    :param min_age minimum possible age
    :param max_age maximum possible age
    :param category: corresponds to a set of two values, representing a gap of ages (a,b), a < b
    :param missing_person_age: number between min and max age, representing the actual age of the remains
    :return:
    """
    mu, sigma = missing_person_age, math.sqrt(obtain_sigma(missing_person_age, min_age))
    normal_distribution = st.NormalDist(mu, sigma)
    upper = normal_distribution.cdf(category[1]) - normal_distribution.cdf(category[0])
    lower = normal_distribution.cdf(max_age) - normal_distribution.cdf(min_age)
    return upper/lower



def obtain_sigma(age, min_age):
    """
    according to age, obtains value of sigma (or variance for normal distribution)
    :param age:
    :return:
    """
    if min_age <= age or age <= 1:
        return 0.5
    elif 1 < age or age <= 15:
        return 2
    else:
        return 5
