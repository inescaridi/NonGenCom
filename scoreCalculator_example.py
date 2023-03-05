import pandas as pd

from nonGenCom.ScoreCalculator import ScoreCalculator
from nonGenCom.Variables.AgeV1 import AgeV1
from nonGenCom.Variables.AgeV2 import AgeV2
from nonGenCom.Variables.BiologicalSex import BiologicalSex

if __name__ == '__main__':
    fc_db = pd.read_csv("tests/resources/Database_FC_proof1_ID.csv")
    mp_db = pd.read_csv("tests/resources/Database_MP_proof1_ID.csv")

    # remove the last parameter for comparing against whole fc database
    score_calculator = ScoreCalculator(fc_db, mp_db, "ID", "ID", ["FC-4", "FC-8"])

    score_calculator.add_score(BiologicalSex(), "Female bias", "High", "FC estimate Biological Sex", "Sex")
    score_calculator.add_score(AgeV1(), "Standard", None, "FC Age Category", "Age")
    score_calculator.add_score(AgeV2(), "Standard", None, "FC Age Category", "Age")

    score_calculator.get_results().to_csv("scoreCalculator_example_output.csv", index=False)
