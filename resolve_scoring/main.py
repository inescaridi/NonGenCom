import pandas as pd
import typer

from dynaconf import settings
from prometheus_client import Summary, CollectorRegistry
from typing import List, Optional

from .models.settings import BiologicalSexSettings
from .scores.bayesian.biological_sex import BiologicalSexScore
from .utils import print_prometheus_report, print_series


app = typer.Typer()

prom_reg = CollectorRegistry()
sum_prep = Summary('prep_processing_seconds', 'Time spent preparing the data', registry=prom_reg)
sum_merge = Summary('merge_processing_seconds', 'Time spent processing merge', registry=prom_reg)
sum_score = Summary('score_processing_seconds', 'Time spent processing score', registry=prom_reg)
sum_sort = Summary('sort_processing_seconds', 'Time spent processing sort', registry=prom_reg)


@app.command()
def score(
    request: str = typer.Option(
        ..., "--req", "-r", help="The data file containing the requests to match."
    ),
    source: str = typer.Option(
        ..., "--src", "-s", help="The data file containing the data to rank."
    ),
    output: str = typer.Option(..., "--out", "-o", help="The result output file."),
    biosex_context: str = typer.Option(
        ..., "--biosex-context", help="The biological sex context."
    ),
    biosex_scenario: str = typer.Option(
        ..., "--biosex-scenario", help="The biological sex scenario."
    ),
    request_ids: Optional[List[str]] = typer.Option(
        None, "--req-id", "-i", help="ID to subset the req."
    ),
):
    with sum_prep.time():
        # Getting biological sex parameters
        biosex_settings: BiologicalSexSettings = BiologicalSexSettings(
            **settings["BIOLOGICAL_SEX"]
        )
        biosex_score: BiologicalSexScore = BiologicalSexScore(
            context=biosex_context, scenario=biosex_scenario, settings=biosex_settings
        )
        print_series(f"Biological sex prior ({biosex_score.context})", biosex_score.prior)
        print_series(f"Biological sex likelihood ({biosex_score.scenario})", biosex_score.likelihood)
        print_series(f"Biological sex posterior", biosex_score.posterior)

        # Getting request data frame
        req_df = pd.read_csv(request)
        # TODO put keys in configuration
        req_df["req_biosex"] = req_df["FC estimate Biological Sex"].map(
            biosex_settings.mapping
        )

        # Getting source data frame
        src_df = pd.read_csv(source)
        # TODO put keys in configuration
        src_df["src_biosex"] = src_df["Sex"].map(biosex_settings.mapping)

    with sum_merge.time():
        # Merging request and source
        merge_df = (
            req_df if not request_ids else req_df[req_df["ID"].isin(request_ids)]
        ).merge(src_df, how="cross", suffixes=("_FC", "_MP"))

    with sum_score.time():
        # Processing score
        scored_df = merge_df.pipe(
            biosex_score, biosex_settings.score_colname, "req_biosex", "src_biosex"
        )
    
    with sum_sort.time():
        # Sorting and outputing
        scored_df.sort_values(biosex_settings.score_colname, ascending=False).to_csv(output)
    
    # prom_report = generate_latest(registry=prom_reg)
    # print(prom_report.decode("utf-8"))
    print_prometheus_report(registry=prom_reg)




@app.callback()
def callback():
    """
    Resolve scoring CLI app.
    """


if __name__ == "__main__":
    app()
