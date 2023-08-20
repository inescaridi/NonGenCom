import pandas as pd

from nonGenCom.Utils import merge_dbs
from nonGenCom.Variables.BiologicalSex import BiologicalSex

if __name__ == '__main__':
    fc_db = pd.read_csv("examples/resources/Database_FC_proof1_ID.csv")
    fc_db[["FC Age", "FC Age Minimum", "FC Age Maximum"]] = fc_db[["FC Age", "FC Age Minimum", "FC Age Maximum"]].astype('Int64')

    mp_db = pd.read_csv("examples/resources/Database_MP_proof1_ID.csv")
    mp_db['Age'] = mp_db['Age'].astype('Int64')

    # FC-SELECTION
    # remove the last parameter for comparing against whole fc database
    # be careful with matching column renames
    merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', ["FC-1024", "FC-1063"])
    # merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', ["FC-4", "FC-149", "FC-233", "FC-354", "FC-573", "FC-991"])

    result = (merged_dbs
              .pipe(BiologicalSex("Uniform", "High", "Perfect representation").add_fc_score, "FC estimate Biological Sex", "Sex")
              )

    result['Final Score T1'] = result[BiologicalSex.SCORE_COLNAME]

    result.to_csv("examples/scoreCalculator_example_fc_select_output.csv", index=False)
