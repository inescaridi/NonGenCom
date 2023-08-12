from unittest import TestCase

from nonGenCom.Variables.Body import Body


class TestBody(TestCase):

    def test_posterior(self):
        body = Body("Standard", "Head & Neck/Disease", "Head & Neck/Disease", "Head & Neck/Disease")

        expected = 0.5  # TODO replace with the real expected value
        obtained = body.get_fc_score()

        self.assertAlmostEqual(expected, obtained, msg="different results")
