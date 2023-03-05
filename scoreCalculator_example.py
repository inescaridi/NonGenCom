import pandas as pd

from nonGenCom.Utils import merge_dbs
from nonGenCom.Variables.AgeV1 import AgeV1
from nonGenCom.Variables.AgeV2 import AgeV2
from nonGenCom.Variables.BiologicalSex import BiologicalSex

if __name__ == '__main__':
    fc_db = pd.read_csv("tests/resources/Database_FC_proof1_ID.csv")
    mp_db = pd.read_csv("tests/resources/Database_MP_proof1_ID.csv")
    mp_db['Age'] = mp_db['Age'].astype('Int64')

    # remove the last parameter for comparing against whole fc database
    # be careful with matching column renames
    merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', ["FC-4"])
    # merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', ["FC-4", "FC-149", "FC-233", "FC-354", "FC-573", "FC-991"])

    result = (merged_dbs
              .pipe(BiologicalSex().add_score_fc_by_merge, "Female bias", "High", "FC estimate Biological Sex", "Sex")
              .pipe(AgeV1().add_score_fc_by_merge, "Standard", None, "FC Age Category", "Age")
              .pipe(AgeV2().add_score_fc_by_apply, "Standard", None, "FC Age Category", "Age")

              )

    result.to_csv("scoreCalculator_example_output.csv", index=False)
