import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

minAge = -1
maxAge = 100


def likelihood_age(forensic_age, missing_person_age):
    """
    :param forensic_age: corresponds to one of the six categories (fetus, infant, child, teenager, adult, elderly) commonly known as x
    :param missing_person_age: possibly, number between min and max age, commonly known as y
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
