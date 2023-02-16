import numpy as np
import pandas as pd

from io import StringIO
from typing import Any

from .bayesian import BayesianScore
from ...models.settings import BiologicalSexSettings

def parse_float(n: Any) -> float:
    if isinstance(n, str):
        n = n.replace(',', '.')
    return float(n)


def convert_all_cells_to_float(df: pd.DataFrame) -> None:
    df[:] = np.vectorize(parse_float)(df)


class BiologicalSexScore(BayesianScore):
    context: str
    scenario: str
    settings: BiologicalSexSettings

    def __init__(self, context: str, scenario: str, settings: BiologicalSexSettings):
        self.context = context
        self.scenario = scenario
        self.settings = settings

        prior = self._get_prior()
        likelihood = self._get_likelihood()

        super().__init__(prior=prior, likelihood=likelihood, l_index="FC", t_index="MP")

    def _get_prior(self):
        context_csv = StringIO(self.settings.context)
        contexts = pd.read_csv(context_csv, skiprows=1, header=None).transpose()
        contexts[0] = contexts[0].str.upper()
        cont_columns = contexts.iloc[0]
        contexts = contexts.drop(index=0).set_axis(cont_columns, axis=1).set_index("MP")
        context = contexts[self.context]
        convert_all_cells_to_float(context)

        return context

    def _get_likelihood(self):
        scenarios_csv = StringIO(self.settings.scenarios)
        scenarios = pd.read_csv(scenarios_csv, skiprows=1, header=None).transpose()
        scenarios[0] = scenarios[0].str.upper()
        scenarios[1] = scenarios[1].str.upper()
        scen_columns = scenarios.iloc[0]
        scenarios = (
            scenarios.drop(index=0)
            .set_axis(scen_columns, axis=1)
            .set_index(["FC", "MP"])
        )

        scenario = scenarios[self.scenario]
        convert_all_cells_to_float(scenario)
        scenario.fillna(0, inplace=True)

        return scenario
