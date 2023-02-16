import statistics as st
import numpy as np

def likelihood_age_two(min_age, max_age):
    category_ranges = []
    for j in range(min_age, max_age - 1):
        category_ranges.append((j, j + 1))
    return likelihood_age(min_age, max_age, category_ranges)


def likelihood_age(min_age, max_age, category_ranges):
    """
    The method computes de conditional probability of a chosen range of ages given it was a particular age
    or more formally:   P(FC = category | MP = missing_person_age)
    :param min_age minimum possible age
    :param max_age maximum possible age
    :param category_ranges: corresponds to a set of two values, representing a gap of ages (a,b), a < b
    :param missing_person_age: number between min and max age, representing the actual age of the remains
    :return:
    """
    matrix_shape = (len(category_ranges), max_age - min_age + 1)
    likelihood_matrix = np.zeros(matrix_shape, float)
    for missing_person_age in range(min_age, max_age):
        mu, sigma = missing_person_age, obtain_sigma(missing_person_age, min_age)
        normal_distribution = st.NormalDist(mu, sigma)
        lower = normal_distribution.cdf(max_age) - normal_distribution.cdf(min_age)

        category_index = 0
        for age_limit in category_ranges:
            upper = normal_distribution.cdf(age_limit[1]) - normal_distribution.cdf(age_limit[0])
            likelihood_matrix[category_index][missing_person_age] = upper / lower
            category_index = category_index + 1

    return likelihood_matrix


def obtain_sigma(age, min_age):
    """
    according to age, obtains value of sigma (or variance for normal distribution)
    :param age
    :param min_age
    :return:
    """
    if min_age <= age or age <= 1:
        return 0.5
    elif 1 < age or age <= 15:
        return 2
    else:
        return 5
