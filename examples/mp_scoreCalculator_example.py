import pandas as pd

from nonGenCom.Utils import merge_dbs
from nonGenCom.Variables.AgeContinuous import AgeContinuous
from nonGenCom.Variables.BiologicalSex import BiologicalSex

if __name__ == '__main__':
    fc_db = pd.read_csv("examples/resources/Database_FC_proof1_ID.csv")
    fc_db[["FC Age", "FC Age Minimum", "FC Age Maximum"]] = fc_db[["FC Age", "FC Age Minimum", "FC Age Maximum"]].astype('Int64')

    mp_db = pd.read_csv("examples/resources/Database_MP_proof1_ID.csv")
    mp_db['Age'] = mp_db['Age'].astype('Int64')

    # MP-SELECTION
    merged_dbs = merge_dbs(mp_db, fc_db, "ID", '_MP', '_FC', ["MP-6"])

    result = (merged_dbs
              .pipe(BiologicalSex().add_score_mp_by_merge, "High", "FC estimate Biological Sex", "Sex")
              .pipe(AgeContinuous(context_name="Standard").add_score_mp_by_apply, "FC Age Minimum", "FC Age Maximum", "Age")
              )

    result['Final Score T1'] = result[BiologicalSex.SCORE_COLNAME] * result[AgeContinuous.SCORE_COLNAME]

    result.sort_values('Final Score T2', ascending=False, inplace=True)

    result.to_csv("examples/scoreCalculator_example_mp_select_output.csv", index=False)
