from unittest import TestCase
from nonGenCom.Age import likelihood_age, likelihood_age_two, obtain_sigma
import statistics as st

class TestAge(TestCase):

    def test_likelihood_age_without_negative_ages(self):
        min_age, max_age = 0, 100
        ranges = [(min_age, 2), (2, 10), (10, 18), (18, 65), (65, max_age)]
        matrix = likelihood_age(min_age, max_age, ranges)

        self.assertEqual(matrix.shape, (len(ranges), 101), "Shape of matrix is incorrect")

        age = 10
        mu, sigma = age, obtain_sigma(age, min_age)
        normal_distribution = st.NormalDist(mu, sigma)
        lower = normal_distribution.cdf(max_age) - normal_distribution.cdf(min_age)
        upper = normal_distribution.cdf(18) - normal_distribution.cdf(10) #teenage

        self.assertEqual(upper/lower, matrix[2][age], "Values obtain are not the same")


    def test_likelihood_age_with_negative_ages(self):
        min_age, max_age = -1, 100
        ranges = [(min_age, 0), (0, 2), (2, 10), (10, 18), (18, 65), (65, max_age)]
        matrix = likelihood_age(min_age, max_age, ranges)

        self.assertEqual(matrix.shape, (len(ranges), 102), "Shape of matrix is incorrect")

        age = 10
        mu, sigma = age, obtain_sigma(age, min_age)
        normal_distribution = st.NormalDist(mu, sigma)
        lower = normal_distribution.cdf(max_age) - normal_distribution.cdf(min_age)
        upper = normal_distribution.cdf(18) - normal_distribution.cdf(10) #teenage

        self.assertEqual(upper/lower, matrix[2][age], "Values obtain are not the same")