import pandas as pd

from nonGenCom.Config import Config
from nonGenCom.ScoreCalculator import ScoreCalculator

if __name__ == '__main__':
    fc_db = pd.read_csv("tests/resources/Database_FC_proof1_ID.csv")
    mp_db = pd.read_csv("tests/resources/Database_MP_proof1_ID.csv")

    result = ScoreCalculator().fc_score_biolsex(fc_db, mp_db, "157,0000", "Female bias", "Challenged",
                                                "FC",
                                                "FC estimate Biological Sex", "Sex")
    result.to_csv("example_output.csv")
