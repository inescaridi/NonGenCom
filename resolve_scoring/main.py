import pandas as pd
import typer

from dynaconf import settings
from typing import List, Optional

from .models.settings import BiologicalSexSettings
from .scores.bayesian.biological_sex import BiologicalSexScore



app = typer.Typer()



@app.command()
def score(
    request: str = typer.Option(
        ..., "--req", "-r", help="The data file containing the requests to match."
    ),
    source: str = typer.Option(
        ..., "--src", "-s", help="The data file containing the data to rank."
    ),
    output: str = typer.Option(
        ..., "--out", "-o", help="The result output file."
    ),
    biosex_context: str = typer.Option(
        ..., "--biosex-context", help="The biological sex context."
    ),
    biosex_scenario: str = typer.Option(
        ..., "--biosex-scenario", help="The biological sex scenario."
    ),
    request_ids: Optional[List[str]] = typer.Option(
        None, "--req-id", "-i", help="ID to subset the req."
    )
):
    # Getting biological sex parameters
    biosex_settings: BiologicalSexSettings = BiologicalSexSettings(**settings["BIOLOGICAL_SEX"])
    biosex_score: BiologicalSexScore = BiologicalSexScore(
            context=biosex_context,
            scenario=biosex_scenario,
            settings=biosex_settings
        )

    # Setting the assignments
    # TODO Only biological sex for now
    assigns = {
        biosex_settings.score_colname: biosex_score,
    }

    # Getting request data frame
    req_df = pd.read_csv(request)
    # TODO put keys in configuration
    req_df["req_biosex"] = req_df["FC estimate Biological Sex"].map(biosex_settings.mapping)

    # Getting source data frame
    src_df = pd.read_csv(source)
    # TODO put keys in configuration
    src_df["src_biosex"] = src_df["Sex"].map(biosex_settings.mapping)

    # Merging request and source
    merge_df = (req_df if not request_ids else req_df[req_df["ID"].isin(request_ids)]).merge(src_df, how='cross', suffixes=('_FC', '_MP'))

    # Processing score
    scored_df = merge_df.assign(**assigns).sort_values(biosex_settings.score_colname, ascending=False)

    scored_df.to_csv(output)



@app.callback()
def callback():
    """
    Resolve scoring CLI app.
    """


if __name__ == "__main__":
    app()
