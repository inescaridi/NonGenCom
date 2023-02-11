import pandas as pd

from nonGenCom.ScoreCalculator import ScoreCalculator

if __name__ == '__main__':
    fc_db = pd.read_csv("tests/resources/Database_FC_proof1_ID.csv")
    mp_db = pd.read_csv("tests/resources/Database_MP_proof1_ID.csv")

    # remove the last parameter for comparing against whole fc database
    result = ScoreCalculator().fc_score_biolsex(fc_db, mp_db, "Female bias", "High", "ID", "FC estimate Biological Sex",
                                                "Sex", ["FC-4", "FC-8"])
    result.to_csv("scoreCalculator_example_output.csv", index=False)
