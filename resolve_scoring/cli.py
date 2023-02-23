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
    
    # Getting request data frame
    req_df = pd.read_csv(request)
    req_df = req_df if not request_ids else req_df[req_df["ID"].isin(request_ids)]

    # Getting source data frame
    src_df = pd.read_csv(source)

    # Getting the score
    scored_df = get_scored_dataframe(
        req_df=req_df,
        src_df=src_df,
        biosex_context=biosex_context,
        biosex_scenario=biosex_scenario,
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
