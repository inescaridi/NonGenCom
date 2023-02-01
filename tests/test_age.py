from unittest import TestCase
from nonGenCom.Age import likelihood_age

class TestAge(TestCase):

    def test_likelihood_age(self):
        likelihood_age(1, 10)