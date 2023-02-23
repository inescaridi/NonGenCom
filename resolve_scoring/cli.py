import pandas as pd
import typer

from typing import List, Optional

from .scores.scores import get_scored_dataframe


app = typer.Typer()


@app.command()
def score(
    request: str = typer.Option(
        ..., "--req", "-r", help="The data file containing the requests to match."
    ),
    request_id_field: str = typer.Option(
        ..., "--req-id-field", help="The ID field of the request data."
    ),
    source: str = typer.Option(
        ..., "--src", "-s", help="The data file containing the data to rank."
    ),
    # source_id_field: str = typer.Option(
    #     ..., "--src-id-field", help="The ID field of the source data."
    # ),
    output: str = typer.Option(..., "--out", "-o", help="The result output file."),
    biosex_context: str = typer.Option(
        ..., "--biosex-context", help="The biological sex context."
    ),
    biosex_scenario: str = typer.Option(
        ..., "--biosex-scenario", help="The biological sex scenario."
    ),
    biosex_req_field: str = typer.Option(
        ..., "--biosex-req-field", help="The biological sex field of the request data."
    ),
    biosex_src_field: str = typer.Option(
        ..., "--biosex-src-field", help="The biological sex field of the source data."
    ),
    request_ids: Optional[List[str]] = typer.Option(
        None, "--req-id", "-i", help="ID to subset the req."
    ),
):
    # Getting request data frame
    req_df = pd.read_csv(request)
    req_df = req_df if not request_ids else req_df[req_df[request_id_field].isin(request_ids)]

    # Getting source data frame
    src_df = pd.read_csv(source)

    # Getting the score
    scored_df = get_scored_dataframe(
        req_df=req_df,
        src_df=src_df,
        biosex_context=biosex_context,
        biosex_scenario=biosex_scenario,
        biosex_req_field=biosex_req_field,
        biosex_src_field=biosex_src_field,
    )

    # Outputting data
    scored_df.to_csv(output)


@app.callback()
def callback():
    """
    Resolve scoring CLI app.
    """


if __name__ == "__main__":
    app()
