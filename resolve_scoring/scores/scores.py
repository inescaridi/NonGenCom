import pandas as pd

from dynaconf import settings
from prometheus_client import Summary, CollectorRegistry
from rich.console import Console

from .bayesian.biological_sex import BiologicalSexScore
from ..models.settings import BiologicalSexSettings
from ..utils import print_prometheus_report, print_series



console = Console()
err_console = Console(stderr=True)



def get_scored_dataframe(
        req_df: pd.DataFrame,
        src_df: pd.DataFrame,
        biosex_context: str,
        biosex_scenario: str,
        prometheus_registry: CollectorRegistry = None
):
    
    if prometheus_registry is None:
        prometheus_registry = CollectorRegistry()
    
    sum_prep = Summary('prep_processing_seconds', 'Time spent preparing the data', registry=prometheus_registry)
    sum_merge = Summary('merge_processing_seconds', 'Time spent processing merge', registry=prometheus_registry)
    sum_score = Summary('score_processing_seconds', 'Time spent processing score', registry=prometheus_registry)
    sum_sort = Summary('sort_processing_seconds', 'Time spent processing sort', registry=prometheus_registry)

    
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

        # TODO put keys in configuration
        req_df["req_biosex"] = req_df["FC estimate Biological Sex"].map(
            biosex_settings.mapping
        )

        # TODO put keys in configuration
        src_df["src_biosex"] = src_df["Sex"].map(biosex_settings.mapping)

    with sum_merge.time():
        # Merging request and source
        merge_df = req_df.merge(src_df, how="cross", suffixes=("_FC", "_MP"))

    with sum_score.time():
        # Processing score
        scored_df = merge_df.pipe(
            biosex_score, biosex_settings.score_colname, "req_biosex", "src_biosex"
        )
    
    with sum_sort.time():
        # Sorting and outputing
        scored_df.sort_values(biosex_settings.score_colname, ascending=False, inplace=True)
    
    print_prometheus_report(registry=prometheus_registry)

    return scored_df