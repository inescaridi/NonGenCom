import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    mu, sigma = missing_person_age, math.sqrt(obtain_sigma(missing_person_age))  # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)
    sns.displot(s)

    plt.show()


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
