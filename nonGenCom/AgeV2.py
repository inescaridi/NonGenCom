from nonGenCom.Age import Age


class AgeV2(Age):
    def __init__(self, min_age: int = -1, max_age: int = 100,
                 contexts_path="nonGenCom/default_inputs/age_contexts.csv",
                 sceneries_path=None,
                 sigmas_path="nonGenCom/default_inputs/age_sigma.csv"):
        super().__init__(contexts_path, sceneries_path, sigmas_path)

        self.min_age = min_age
        self.min_age = max_age

        category_ranges = {}
        for i in range(self.min_age, self.max_age - 1):
            category_ranges[f"Category_{i}"] = (i, i + 1)
        self.category_ranges = category_ranges

        self.version_name = 'V2'
