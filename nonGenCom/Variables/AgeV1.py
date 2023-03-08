from nonGenCom.Variables.Age import Age
from nonGenCom.Utils import load_fc_indexed_file, FC_INDEX_NAME


class AgeV1(Age):
    SCORE_COLNAME = 'age_v1_score'

    def __init__(self, category_ranges_path="nonGenCom/default_inputs/age_ranges.csv",
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv", sceneries_path=None,
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(contexts_path, sceneries_path, sigmas_path, category_ranges_path)


        ranges_df = load_fc_indexed_file(category_ranges_path)
        self.category_ranges = ranges_df.groupby(FC_INDEX_NAME).agg({'age': (min, max)})['age'].apply(tuple, axis=1).to_dict()

        self.min_age = int(ranges_df.age.min())
        self.max_age = int(ranges_df.age.max())

        self.version_name = 'V1'
