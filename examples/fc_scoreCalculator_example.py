import os

import pandas as pd

from nonGenCom.Utils import merge_dbs
from nonGenCom.Variables.BiologicalSex import BiologicalSex

if __name__ == '__main__':
    fc_db = pd.read_csv("examples/resources/FC_database_example.csv")
    mp_db = pd.read_csv("examples/resources/MP_database_example.csv")

    # FC-SELECTION
    # remove the last parameter for comparing against whole fc database
    # be careful with matching column renames
    merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP')
    # merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', [158, 6])
    # merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', [2, 3, 6, 9, 110, 40, 32])

    result = (merged_dbs
              .pipe(BiologicalSex("Uniform", "High", "Perfect representation").add_fc_score, "FC estimate Biological Sex", "MP estimate Biological Sex")
              )

    print(result)

    result.to_csv("examples/results/fc_scoreCalculator_example.csv", index=False)
