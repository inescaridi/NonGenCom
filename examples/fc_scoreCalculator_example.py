import pandas as pd

from nonGenCom.Utils import merge_dbs
from nonGenCom.Variables.Age import Age
from nonGenCom.Variables.BiologicalSex import BiologicalSex
from nonGenCom.Variables.Body import Body
from nonGenCom.Variables.Date import Date
from nonGenCom.Variables.Height import Height

if __name__ == '__main__':
    fc_db = pd.read_csv("examples/resources/FC_database_example.csv")
    mp_db = pd.read_csv("examples/resources/MP_database_example.csv")

    # FC-SELECTION
    # remove the last parameter for comparing against whole fc database
    # be careful with matching column renames
    merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP')
    # merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', [158, 6])
    # merged_dbs = merge_dbs(fc_db, mp_db, "ID", '_FC', '_MP', [2, 3, 6, 9, 110, 40, 32])

    # for each variable we set up an instance, and apply the add_fc_score method
    # we need to pass the column names of the values for each variable in each database
    # for Categorical values we only need the FC value and MP value
    # for Continuous values we need the FC min and max value, and the MP min and max value since we are comparing a range
    result = (merged_dbs
              .pipe(BiologicalSex("Uniform", "High", "Perfect representation").add_fc_score, "FC estimate Biological Sex", "MP estimate Biological Sex")
              .pipe(Age().add_fc_score, "FC estimate Age Minimum (years)", "FC estimate Age Maximum (years)", "MP estimate Age Minimum (years)", "MP estimate Age Maximum (years)")
              .pipe(Date("2020-02-01", "2022-12-01", 130).add_fc_score, "FC Estimated Date of Death", "FC Estimated Date of Burial", "MP Estimated Date of Death", "MP Estimated Date of Burial")
              .pipe(Body("Uniform", "Standard", "Standard", "Torso/Disease").add_fc_score, "Torso/Disease", "MP Torso/Disease")
              .pipe(Height().add_fc_score, "FC height min estimated", "FC height max estimated", "MP height min estimated", "MP height max estimated")
              )

    # allow pandas to print all the columns
    pd.set_option('display.max_columns', None)

    print(result)

    result.to_csv("examples/results/fc_scoreCalculator_example.csv", index=False)
